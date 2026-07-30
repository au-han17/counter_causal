"""
Benchmark task runners for "Back from the Future: Key-Value Cache Management by Counter-Causal Surprise."
Copyright (c) 2026, Metacognition (http://metacognitionai.com)

Supported tasks
---------------
run_math500_task    — MATH500 (Hendrycks et al., 2021)
run_longhealth_task — LongHealth (Kuhnel et al., 2024)
run_qasper_task     — QASPER (Dasigi et al., 2021)
run_locomo_task     — LoCoMo (Maharana et al., 2024)
"""

import re
import json
import string
from collections import Counter
from typing import Optional

from datasets import load_dataset
from math_verify import parse, verify

from hooks import generate_with_kv_hook


# ---------------------------------------------------------------------------
# MATH500
# ---------------------------------------------------------------------------

MATH500_SYSTEM = (
    "Solve the following math problem. Show your reasoning step by step, "
    "then put your final answer in \\boxed{{}}.\n\n"
)

_MATH500_PROMPT = MATH500_SYSTEM + "Problem: {problem}\n\nSolution:"


def build_math500_prompt(sample: dict) -> str:
    return _MATH500_PROMPT.format(problem=sample["problem"])


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in the model output."""
    matches = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not matches:
        return None
    start = matches[-1] + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def compare_math(pred: Optional[str], gold: str) -> bool:
    """Symbolic math comparison via math_verify."""
    if pred is None:
        return False
    return verify(gold=parse(f"${gold}$"), target=parse(f"${pred}$"))


def run_math500_task(model, tokenizer, kv_hook=None, subject=None, level=None,
                     max_new_tokens=2048, chunk_size=256, sample_index=None,
                     refresh_mode='chunked'):
    """
    Evaluate on MATH500 (HuggingFaceH4/MATH-500).

    Returns:
        results:  list of per-sample dicts with pred, answer, correct, etc.
        accuracy: fraction correct.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if subject:
        dataset = dataset.filter(lambda x: x["subject"] == subject)
    if level:
        dataset = dataset.filter(lambda x: x["level"] == level)

    results = []
    correct = 0

    for i, sample in enumerate(dataset):
        if sample_index is not None and i != sample_index:
            continue

        prompt = build_math500_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        def boxed_done(tokens):
            return extract_boxed_answer(tokenizer.decode(tokens, skip_special_tokens=True)) is not None

        raw_pred = generate_with_kv_hook(
            model, tokenizer, inputs.input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            kv_hook=kv_hook,
            stop_fn=boxed_done,
            refresh_mode=refresh_mode,
        )

        pred = extract_boxed_answer(raw_pred)
        gold = sample["answer"]
        is_correct = compare_math(pred, gold)
        if is_correct:
            correct += 1

        results.append({
            "id": sample["unique_id"],
            "subject": sample["subject"],
            "level": sample["level"],
            "pred_raw": raw_pred,
            "pred": pred,
            "answer": gold,
            "correct": is_correct,
        })
        print(f"[{i+1}/{len(dataset)}] pred={pred!r} gold={gold!r} {'✓' if is_correct else '✗'}")

    accuracy = correct / len(results) if results else 0.0
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.3f}")
    return results, accuracy


# ---------------------------------------------------------------------------
# LongHealth
# ---------------------------------------------------------------------------

LONGHEALTH_SYSTEM = (
    "Read the following patient records and answer the multiple-choice question by responding "
    "with only the letter of the correct answer (A, B, C, D, or E).\n\n"
)

_LONGHEALTH_PROMPT = LONGHEALTH_SYSTEM + (
    "Patient Records:\n{context}\n\n"
    "Question: {question}\n\n"
    "A) {answer_a}\nB) {answer_b}\nC) {answer_c}\nD) {answer_d}\nE) {answer_e}\n\n"
    "Answer:"
)

LONGHEALTH_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"


def _build_longhealth_prompt(context: str, q: dict) -> str:
    return _LONGHEALTH_PROMPT.format(context=context, **{
        k: q[k] for k in ("question", "answer_a", "answer_b", "answer_c", "answer_d", "answer_e")
    })


def _longhealth_correct_letter(q: dict) -> Optional[str]:
    for letter, key in zip("ABCDE", ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]):
        if q[key] == q["correct"]:
            return letter
    return None


def _extract_letter(text: str, valid: str = "ABCDE") -> Optional[str]:
    text = text.strip()
    if text and text[0] in valid:
        return text[0]
    for ch in text:
        if ch in valid:
            return ch
    return None


def run_longhealth_task(model, tokenizer, kv_hook=None, max_new_tokens=16,
                        chunk_size=4096, sample_index=None, refresh_mode='chunked'):
    """
    Evaluate on LongHealth (fetched from GitHub).

    Returns:
        results:  list of per-sample dicts.
        accuracy: fraction correct.
    """
    import requests
    data = requests.get(LONGHEALTH_URL).json()

    samples = [
        (pid, "\n\n".join(pd["texts"].values()), q)
        for pid, pd in data.items()
        for q in pd["questions"]
    ]

    results = []
    correct = 0

    for i, (patient_id, context, q) in enumerate(samples):
        if sample_index is not None and i != sample_index:
            continue

        prompt = _build_longhealth_prompt(context, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_pred = generate_with_kv_hook(
            model, tokenizer, inputs.input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            kv_hook=kv_hook,
            refresh_mode=refresh_mode,
        )
        pred = _extract_letter(raw_pred)
        gold = _longhealth_correct_letter(q)
        is_correct = pred == gold
        if is_correct:
            correct += 1

        results.append({
            "patient_id": patient_id,
            "question_no": q["No"],
            "pred_raw": raw_pred,
            "pred": pred,
            "answer": gold,
            "correct": is_correct,
        })
        print(f"[{i+1}/{len(samples)}] pred={pred} gold={gold} {'✓' if is_correct else '✗'}  raw={raw_pred!r}")

    accuracy = correct / len(results) if results else 0.0
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.3f}")
    return results, accuracy


# ---------------------------------------------------------------------------
# QASPER
# ---------------------------------------------------------------------------

QASPER_SYSTEM = (
    "You are an expert researcher. Read the following scientific paper carefully. "
    "Answer the user's question based only on the provided text. Responses can be "
    "extractive, abstractive or yes/no.\n\n"
    "1. Identify the relevant paragraphs, tables, or figures (Evidence).\n"
    "2. Formulate an answer based on the evidence.\n"
    "3. If the answer cannot be found in the text, respond with \"Unanswerable\".\n\n"
)

_QASPER_PROMPT = QASPER_SYSTEM + (
    "Title: {title}\n\nAbstract:\n{abstract}\n\n{sections}Question: {question}\n\nAnswer:"
)

QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
_QASPER_DEV_FILE = "qasper-dev-v0.3.json"


def _build_qasper_context(paper: dict) -> str:
    parts = []
    for section in paper["full_text"]:
        name = section["section_name"]
        paras = section["paragraphs"]
        parts.append((f"{name}:\n" if name else "") + "\n".join(paras))
    return "\n\n".join(parts) + "\n\n" if parts else ""


def _build_qasper_prompt(paper: dict, question: str) -> str:
    return _QASPER_PROMPT.format(
        title=paper["title"], abstract=paper["abstract"],
        sections=_build_qasper_context(paper), question=question,
    )


def _normalize_qasper(text: str) -> list:
    text = text.strip().lower().translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if tokens and tokens[0] in ("unanswerable", "yes", "no"):
        return [tokens[0]]
    return [t for t in tokens if t not in {"a", "an", "the"}]


def _token_f1(pred_tokens: list, gold_tokens: list) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    p = Counter(pred_tokens)
    g = Counter(gold_tokens)
    common = sum(min(p[t], g[t]) for t in g)
    return 2 * common / (len(pred_tokens) + len(gold_tokens)) if common else 0.0


def _get_qasper_gold_answers(answer_annotations: list) -> list:
    gold = []
    for ann in answer_annotations:
        a = ann["answer"]
        if a["unanswerable"]:
            gold.append("unanswerable")
        elif a["yes_no"] is True:
            gold.append("yes")
        elif a["yes_no"] is False:
            gold.append("no")
        elif a["extractive_spans"]:
            gold.append(" ".join(a["extractive_spans"]))
        elif a["free_form_answer"]:
            gold.append(a["free_form_answer"])
        else:
            gold.append("unanswerable")
    return gold


def _score_qasper(pred: str, gold_answers: list) -> float:
    pred_toks = _normalize_qasper(pred)
    return max(_token_f1(pred_toks, _normalize_qasper(g)) for g in gold_answers)


def run_qasper_task(model, tokenizer, kv_hook=None, max_new_tokens=128,
                    chunk_size=4096, sample_index=None, refresh_mode='chunked'):
    """
    Evaluate on QASPER dev set (downloaded from S3).

    Returns:
        results: list of per-sample dicts with f1, pred_raw, gold_answers.
        mean_f1: mean token F1 across all questions.
    """
    import requests, tarfile, io
    print("Downloading QASPER ...")
    response = requests.get(QASPER_URL)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        raw_data = json.load(tar.extractfile(_QASPER_DEV_FILE))

    samples = [
        (pid, {**paper, "id": pid}, qa["question"], qa["question_id"], qa["answers"])
        for pid, paper in raw_data.items()
        for qa in paper["qas"]
    ]

    results = []
    total_f1 = 0.0

    for i, (paper_id, paper, question, question_id, answer_annotations) in enumerate(samples):
        if sample_index is not None and i != sample_index:
            continue

        prompt = _build_qasper_prompt(paper, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_pred = generate_with_kv_hook(
            model, tokenizer, inputs.input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            kv_hook=kv_hook,
            refresh_mode=refresh_mode,
        )
        gold_answers = _get_qasper_gold_answers(answer_annotations)
        f1 = _score_qasper(raw_pred, gold_answers)
        total_f1 += f1

        results.append({
            "paper_id": paper_id,
            "question_id": question_id,
            "question": question,
            "pred_raw": raw_pred,
            "gold_answers": gold_answers,
            "f1": f1,
        })
        print(f"[{i+1}/{len(samples)}] f1={f1:.3f}  pred={raw_pred[:80]!r}  gold={gold_answers[0]!r}")

    mean_f1 = total_f1 / len(results) if results else 0.0
    print(f"\nMean Token F1: {total_f1:.3f}/{len(results)} = {mean_f1:.3f}")
    return results, mean_f1


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

LOCOMO_SYSTEM = (
    "You are an AI assistant tasked with analyzing a conversation between {speaker1} and {speaker2}. "
    "Based on the provided conversation sessions, answer the question accurately. "
    "Focus on recalling past facts, user preferences, and temporal relationships. "
    "Answer the question using exact words from the conversation when possible.\n\n"
)

_LOCOMO_PROMPT = LOCOMO_SYSTEM + "Input: {conversation}\n\nQuestion: {question}\nAnswer:"

_LOCOMO_ADVERSARIAL_PROMPT = LOCOMO_SYSTEM + (
    "Input: {conversation}\n\n"
    "If the answer cannot be found in the conversation, respond with \"no information available\".\n"
    "Question: {question}\nAnswer:"
)

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def _build_locomo_conversation(conversation: dict):
    """Format all sessions into a flat string; return (text, speaker1, speaker2)."""
    session_keys = sorted(
        [k for k in conversation if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )
    speakers_seen = []
    parts = []
    for sk in session_keys:
        date_str = conversation.get(f"{sk}_date_time", "")
        dialogs = conversation[sk]
        if not isinstance(dialogs, list):
            continue
        if date_str:
            parts.append(f"[{date_str}]")
        for dialog in dialogs:
            speaker = dialog.get("speaker", "")
            text = dialog.get("text", "")
            caption = dialog.get("blip_caption", "")
            if speaker and speaker not in speakers_seen:
                speakers_seen.append(speaker)
            line = f"{speaker}: {text}" if speaker else text
            if caption:
                line += f" [shared image: {caption}]"
            parts.append(line)
    speaker1 = speakers_seen[0] if len(speakers_seen) > 0 else "Person A"
    speaker2 = speakers_seen[1] if len(speakers_seen) > 1 else "Person B"
    return "\n".join(parts), speaker1, speaker2


try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    def _stem(w): return _stemmer.stem(w)
except ImportError:
    def _stem(w): return w


def _normalize_locomo(text: str) -> list:
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return [_stem(t) for t in text.split() if t not in {"a", "an", "the", "and"}]


def _locomo_token_f1(pred_tokens: list, gold_tokens: list) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = sum(min(pred_c[t], gold_c[t]) for t in gold_c)
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_locomo(pred: str, answer: str, category: int) -> float:
    pred = str(pred).strip()
    answer = str(answer)
    if category == 5:
        lower = pred.lower()
        return 1.0 if ("no information available" in lower or "not mentioned" in lower) else 0.0
    if category == 1:
        gold_parts = [p.strip() for p in answer.split(",") if p.strip()]
        pred_parts = [p.strip() for p in pred.split(",") if p.strip()] or [pred]
        if not gold_parts:
            return 0.0
        return sum(
            max((_locomo_token_f1(_normalize_locomo(p), _normalize_locomo(g)) for p in pred_parts), default=0.0)
            for g in gold_parts
        ) / len(gold_parts)
    return _locomo_token_f1(_normalize_locomo(pred), _normalize_locomo(answer))


def _load_locomo(data_path=None):
    if data_path:
        with open(data_path) as f:
            data = json.load(f)
    else:
        import requests
        print(f"Downloading LoCoMo from {LOCOMO_URL} ...")
        data = requests.get(LOCOMO_URL, timeout=120).json()
    items = data.values() if isinstance(data, dict) else data
    return [(item.get("conversation", {}), qa) for item in items for qa in item.get("qa", [])]


def run_locomo_task(model, tokenizer, kv_hook=None, category=None, max_new_tokens=32,
                    chunk_size=4096, sample_index=None, data_path=None, max_samples=None,
                    refresh_mode='chunked'):
    """
    Evaluate on LoCoMo (10-conversation subset, fetched from GitHub).

    Args:
        category:    If set, restrict to QA pairs of that category (1–5).
                     Category 5 is adversarial (answer is "no information available").
        data_path:   Local path to locomo10.json; downloads from GitHub if None.
        max_samples: Cap the number of samples evaluated (useful for quick checks).

    Returns:
        results:    list of per-sample dicts with category, score, pred_raw, answer.
        mean_score: mean F1 across all evaluated samples.
        per_cat:    dict mapping category id -> mean F1 for that category.
    """
    samples = _load_locomo(data_path)
    if category is not None:
        samples = [(c, q) for c, q in samples if q.get("category") == category]

    results = []
    category_scores = {}
    global_score = 0.0

    for i, (conversation, qa) in enumerate(samples):
        if sample_index is not None and i != sample_index:
            continue
        if max_samples is not None and len(results) >= max_samples:
            break

        conv_text, speaker1, speaker2 = _build_locomo_conversation(conversation)
        cat = qa.get("category", 1)
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        template = _LOCOMO_ADVERSARIAL_PROMPT if cat == 5 else _LOCOMO_PROMPT
        prompt = template.format(
            speaker1=speaker1, speaker2=speaker2,
            conversation=conv_text, question=question,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_pred = generate_with_kv_hook(
            model, tokenizer, inputs.input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            kv_hook=kv_hook,
            refresh_mode=refresh_mode,
        )

        sc = _score_locomo(raw_pred, answer, cat)
        global_score += sc
        category_scores.setdefault(cat, []).append(sc)

        results.append({
            "idx": i,
            "category": cat,
            "question": question,
            "pred_raw": raw_pred,
            "answer": answer,
            "score": sc,
        })
        print(f"[{len(results)}/{len(samples)}] cat={cat} score={sc:.3f}  pred={raw_pred[:80]!r}")

    mean_score = global_score / len(results) if results else 0.0
    per_cat = {cat: sum(v) / len(v) for cat, v in category_scores.items()}
    print(f"\nMean F1: {mean_score:.3f}")
    for cat in sorted(per_cat):
        print(f"  cat {cat}: {per_cat[cat]:.3f}  (n={len(category_scores[cat])})")
    return results, mean_score, per_cat
