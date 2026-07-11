"""Shared utilities for the automatic-prompt-optimization (ape-*) MLS-Bench tasks.

This package hosts a genuinely NEW LLM direction: AUTOMATIC PROMPT OPTIMIZATION
(APE, Zhou et al. 2022; Honovich et al. 2022) — searching over the INSTRUCTION
TEXT of a prompt to maximize task accuracy of a FROZEN small instruction LM. It is
DISTINCT from in-context-learning (the icl-* tasks), which selects/orders/calibrates
the DEMONSTRATIONS while the instruction/prompt template is held FIXED. Here it is
the OPPOSITE: the demonstrations are FIXED/ABSENT (zero-shot execution) and the agent
designs how to PROPOSE and SELECT the instruction string.

Fixed, un-gameable evaluation path:

  * A single FROZEN small instruction LM (Qwen2.5-0.5B-Instruct) is loaded once.
  * For each task there is a FIXED, forced-choice classification EXECUTOR: given an
    INSTRUCTION string and an input, the LM predicts a label by argmax length-
    normalized log-prob over the FIXED label set (no free generation, no drift —
    a valid label always comes out). The per-label score is CONTEXTUALLY CALIBRATED
    (Zhao et al. 2021, "Calibrate Before Use"): the content-free label distribution
    under the same instruction (input = "N/A") is subtracted, so the prediction
    reflects how the real input shifts each label ABOVE its baseline rather than the
    LM's / instruction's label-prior bias. Without this a small LM's forced-choice
    scores are dominated by label-word frequency and by label words that leak into
    the instruction text, and instruction quality is NOT monotone; with it, an
    empty/degenerate instruction collapses toward the class prior and a genuine task
    description lifts accuracy. This executor is FIXED; the agent never touches it.
  * Each dataset is split into THREE DISJOINT parts (deterministic, seed 42):
      - a PROPOSAL few-shot pool (labeled input/output examples the agent may show
        the LM to have it INDUCE candidate instructions, a la reverse-mode APE),
      - a small DEV set (used to SCORE/SELECT candidate instructions — the search
        signal), and
      - a HELD-OUT TEST set (the SCORED metric; the agent NEVER sees test labels).
  * The agent controls ONLY the APE policy for its task:
      - ape-instruction-search : how to PROPOSE candidates (fixed generic vs LM-
        induced from the pool vs iterative resampling) + SELECT one on dev.
      - ape-candidate-scoring  : the ESTIMATOR that ranks a FIXED candidate pool
        (random vs dev exec-accuracy vs answer log-likelihood).
      - ape-search-strategy    : the SELECTION/allocation over a LIMITED dev-eval
        budget (pick-first / overfit-tiny-dev vs robust generalizing selection).

Because selection uses only DEV but the score is on the DISJOINT TEST set, an
instruction overfit to a tiny dev slice does NOT win, and an empty/degenerate
instruction collapses toward the class prior — the metric is monotone in true
instruction quality.

Model + data are staged offline under $APE_MODEL / $APE_DATA so the container runs
with HF_HUB_OFFLINE=1. Everything is deterministic (seed 42). Inference-only.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


PROTOCOL = "ape_qwen25_05b_full_official_v1"
EXPECTED_EVAL = {"agnews": 7_600, "sst2": 872}
EXPECTED_POOL = 128
EXPECTED_DEV = 200


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Paths / determinism / surface loading
# ---------------------------------------------------------------------------
def model_path() -> str:
    """FROZEN small instruction LM. All ape-* tasks execute from this model."""
    return os.environ.get(
        "APE_MODEL", "/data/prompt-optimization/models/Qwen2.5-0.5B-Instruct"
    )


def data_root() -> Path:
    return Path(os.environ.get("APE_DATA", "/data/prompt-optimization/data"))


def set_seeds(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_surface(sol_path: str, attr: str):
    """Import the agent-editable callable `attr` from solution/<file>.py."""
    p = Path(sol_path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, attr):
        raise SystemExit(f"solution must define `{attr}(...)`")
    return getattr(mod, attr)


# ---------------------------------------------------------------------------
# Datasets (FIXED proposal pool + DEV + disjoint TEST)
# ---------------------------------------------------------------------------
class Dataset:
    """A frozen APE dataset: labeled proposal POOL + DEV + disjoint TEST + verbalizer.

    Attributes:
      pool     list[dict(text,label)]  — few-shot examples for instruction PROPOSAL
      dev      list[dict(text,label)]  — small set to SCORE/SELECT candidates
      test     list[dict(text,label)]  — HELD-OUT set (the scored accuracy target)
      labels   dict[int,str]           — label id -> verbalizer word (fixed)
      n_class  int
      task     str                     — "topic" | "sentiment"
    """

    def __init__(self, name: str):
        d = data_root() / name
        manifest_path = data_root() / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("canonical APE data manifest is missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol") != PROTOCOL:
            raise RuntimeError("unexpected APE data protocol")
        protocol_sha = manifest.get("protocol_sha256")
        unhashed_manifest = dict(manifest)
        unhashed_manifest.pop("protocol_sha256", None)
        if protocol_sha != _canonical_sha256(unhashed_manifest):
            raise RuntimeError("APE protocol manifest digest does not match")
        records = manifest.get("datasets", {})
        if name not in records:
            raise RuntimeError(f"dataset {name!r} is absent from the manifest")
        record = records[name]
        data_sha = record.get("data_sha256")
        unhashed_record = dict(record)
        unhashed_record.pop("data_sha256", None)
        if data_sha != _canonical_sha256(unhashed_record):
            raise RuntimeError(f"{name} data manifest digest does not match")
        self.name = name
        self.pool = json.loads((d / "pool.json").read_text())
        self.dev = json.loads((d / "dev.json").read_text())
        self.test = json.loads((d / "test.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        self.labels = {int(k): v for k, v in meta["labels"].items()}
        self.n_class = int(meta["n_class"])
        self.task = meta["task"]
        self.evaluation_split = str(meta["evaluation_split"])
        self.protocol_sha256 = str(protocol_sha)
        self.data_sha256 = str(data_sha)
        self.model_sha256 = str(manifest.get("model", {}).get("model_sha256", ""))

        expected_split = "test" if name == "agnews" else "validation"
        expected = EXPECTED_EVAL.get(name)
        if (
            len(self.pool) != EXPECTED_POOL
            or len(self.dev) != EXPECTED_DEV
            or expected is None
            or len(self.test) != expected
            or self.evaluation_split != expected_split
            or record.get("pool_n") != EXPECTED_POOL
            or record.get("dev_n") != EXPECTED_DEV
            or record.get("eval_n") != expected
            or record.get("pairwise_text_disjoint") is not True
        ):
            raise RuntimeError(f"{name} protocol inventory is incomplete")

        for filename in ("pool.json", "dev.json", "test.json", "meta.json"):
            payload = (d / filename).read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            if record.get("files", {}).get(filename) != actual:
                raise RuntimeError(f"{name}/{filename} does not match its manifest")

        text_sets = [
            {str(row["text"]) for row in rows}
            for rows in (self.pool, self.dev, self.test)
        ]
        if (
            len(text_sets[0]) != EXPECTED_POOL
            or len(text_sets[1]) != EXPECTED_DEV
            or text_sets[0] & text_sets[1]
            or text_sets[0] & text_sets[2]
            or text_sets[1] & text_sets[2]
        ):
            raise RuntimeError(f"{name} proposal/selection/evaluation texts overlap")
        if any(
            not isinstance(row.get("label"), int)
            or not 0 <= row["label"] < self.n_class
            or not isinstance(row.get("text"), str)
            or not row["text"]
            for rows in (self.pool, self.dev, self.test)
            for row in rows
        ):
            raise RuntimeError(f"{name} contains a malformed example")

    def label_words(self):
        return [self.labels[i] for i in range(self.n_class)]

    def input_field(self, text: str) -> str:
        """How ONE input is rendered under an instruction (FIXED). The label word
        is appended by the executor via forced-choice scoring, not generated."""
        if self.task == "sentiment":
            return f"Review: {text}\nAnswer:"
        return f"News: {text}\nAnswer:"


def load_dataset(name: str) -> Dataset:
    return Dataset(name)


# ---------------------------------------------------------------------------
# FROZEN instruction LM + FIXED forced-choice executor
# ---------------------------------------------------------------------------
class Executor:
    """Frozen instruction LM with a FIXED forced-choice classification executor.

    Given an INSTRUCTION string and a batch of inputs, predicts each input's label
    as the label word the LM assigns the highest length-normalized log-prob under
    the FIXED prompt assembly:  <chat-template>[ instruction + input_field ] label.

    This engine is FIXED — the agent controls ONLY the instruction string(s) fed in
    (and, per task, how candidates are proposed/scored/selected). A degenerate/empty
    instruction collapses toward the class prior; a good instruction lifts accuracy.

    Every call is COUNTED (n_exec_calls) so the search-strategy task can enforce a
    dev-evaluation budget.
    """

    def __init__(self, ds: Dataset, seed: int = 42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        set_seeds(seed)
        self.ds = ds
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path())
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_path(), torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.label_ids = [
            self.tok(" " + w, add_special_tokens=False)["input_ids"]
            for w in ds.label_words()
        ]
        self.n_exec_calls = 0            # number of (instruction,input) executions
        self._cache: dict = {}           # (instruction, text_idx_key) -> logprobs
        self._cal_cache: dict = {}       # instruction -> content-free label logprobs
        self.batch_size = max(1, int(os.environ.get("APE_BATCH_SIZE", "16")))
        self.selected_test_passes = 0
        self.last_test_correct: int | None = None

    # -- prompt assembly (FIXED) ------------------------------------------------
    def _build_user_msg(self, instruction: str, text: str) -> str:
        instr = (instruction or "").strip()
        body = self.ds.input_field(text)
        return (instr + "\n\n" + body) if instr else body

    def _encode_ids(self, user_msg: str) -> list[int]:
        s = self.tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(self.tok(s, add_special_tokens=False)["input_ids"])
        if not ids:
            raise RuntimeError("chat-template tokenization produced an empty prompt")
        return ids

    def _encode(self, user_msg: str):
        return self.torch.tensor(
            [self._encode_ids(user_msg)], dtype=self.torch.long, device=self.device
        )

    def _raw_label_logprobs(self, user_msg: str) -> np.ndarray:
        """Length-normalized log-prob the LM assigns each label word as the
        continuation of the assistant turn for a fully-assembled user message."""
        return self._raw_label_logprobs_many([user_msg])[0]

    def _raw_label_logprobs_many(self, user_msgs: list[str]) -> np.ndarray:
        """Batched label scoring for a list of fully assembled user messages."""
        torch = self.torch
        out = np.zeros((len(user_msgs), self.ds.n_class), dtype=np.float64)
        cases: list[tuple[int, int, list[int], list[int]]] = []
        for message_index, message in enumerate(user_msgs):
            base_ids = self._encode_ids(message)
            if not base_ids:
                raise RuntimeError("tokenizer produced an empty prompt")
            for label_index, label_ids in enumerate(self.label_ids):
                cases.append(
                    (message_index, label_index, base_ids, list(label_ids))
                )

        for start in range(0, len(cases), self.batch_size):
            batch = cases[start:start + self.batch_size]
            sequences = [base + label for _, _, base, label in batch]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = torch.full(
                (len(batch), max_len),
                int(self.tok.pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                input_ids[row_index, :len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=self.device
                )
                attention_mask[row_index, :len(sequence)] = 1

            with torch.inference_mode():
                logits = self.lm(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            for row_index, (message_index, label_index, base, label) in enumerate(batch):
                positions = torch.arange(
                    len(base) - 1,
                    len(base) - 1 + len(label),
                    device=self.device,
                )
                step_logits = logits[row_index, positions].float()
                token_ids = torch.tensor(label, dtype=torch.long, device=self.device)
                token_logprobs = torch.log_softmax(step_logits, dim=-1).gather(
                    1, token_ids[:, None]
                )[:, 0]
                out[message_index, label_index] = float(token_logprobs.mean().item())
            del logits, input_ids, attention_mask
        return out

    def _calibration(self, instruction: str) -> np.ndarray:
        """Content-free label log-probs for this instruction (Zhao et al. 2021,
        "Calibrate Before Use"): the label distribution the LM emits under the
        SAME instruction but a content-free input ("N/A"). Subtracting this removes
        the instruction's / label-word-leakage's PRIOR bias so the prediction
        reflects how the actual input shifts each label ABOVE its baseline. FIXED,
        computed once per instruction (its cost is NOT charged to the search budget).
        """
        c = self._cal_cache.get(instruction)
        if c is None:
            c = self._raw_label_logprobs(self._build_user_msg(instruction, "N/A"))
            self._cal_cache[instruction] = c
        return c

    def label_logprobs_one(self, instruction: str, text: str) -> np.ndarray:
        """CALIBRATED length-normalized label log-probs (FIXED scoring): the raw
        per-label log-prob minus the content-free calibration for this instruction.
        Returns [n_class]. A degenerate/empty instruction leaves the calibrated
        scores near uniform (accuracy near the class prior); an instruction that
        genuinely directs the LM to the task lifts the correct label above its
        content-free baseline."""
        raw = self._raw_label_logprobs(self._build_user_msg(instruction, text))
        return raw - self._calibration(instruction)

    def predict(self, instruction: str, rows) -> np.ndarray:
        """Predicted label id for each row under `instruction`. Counts executions.
        Cached per (instruction, row text) so repeated dev evals of the same
        candidate are free (the agent's search budget is measured in UNIQUE
        (instruction,input) executions)."""
        self._ensure_logprobs(instruction, rows)
        return np.asarray(
            [int(np.argmax(self._cache[(instruction, row["text"])])) for row in rows],
            dtype=np.int64,
        )

    def _ensure_logprobs(self, instruction: str, rows) -> None:
        missing: list[str] = []
        seen = set()
        for row in rows:
            text = row["text"]
            key = (instruction, text)
            if key not in self._cache and text not in seen:
                seen.add(text)
                missing.append(text)
        if not missing:
            return
        messages = [self._build_user_msg(instruction, text) for text in missing]
        raw = self._raw_label_logprobs_many(messages)
        calibration = self._calibration(instruction)
        for text, values in zip(missing, raw):
            self._cache[(instruction, text)] = values - calibration
        self.n_exec_calls += len(missing)

    def label_logprob_matrix(self, instruction: str, rows) -> np.ndarray:
        """[n_rows, n_class] label log-probs under `instruction` (for estimators
        that use log-likelihood rather than argmax accuracy). Counts executions."""
        self._ensure_logprobs(instruction, rows)
        return np.stack(
            [self._cache[(instruction, row["text"])] for row in rows], axis=0
        )

    def dev_accuracy(self, instruction: str, dev_rows) -> float:
        preds = self.predict(instruction, dev_rows)
        return float(np.mean([int(p == r["label"])
                              for p, r in zip(preds, dev_rows)]))


# ---------------------------------------------------------------------------
# Instruction PROPOSAL via the frozen LM (reverse-mode induction) — a helper the
# search/scoring surfaces may use. FIXED machinery (deterministic greedy decode).
# ---------------------------------------------------------------------------
def induce_instructions(executor: "Executor", pool_rows, n_candidates: int,
                        seed: int = 42) -> list[str]:
    """Ask the frozen LM to INDUCE candidate task instructions from labeled
    input/output examples (Honovich/Zhou reverse-mode APE). Returns up to
    n_candidates distinct instruction strings. Deterministic: different few-shot
    example subsets (seeded) yield different candidates. This is a convenience
    proposer; a surface may also hand-write candidates or resample."""
    import torch
    ds = executor.ds
    lab = ds.labels
    rng = random.Random(seed)
    tok, lm, device = executor.tok, executor.lm, executor.device
    cands: list[str] = []
    seen = set()
    attempts = 0
    while len(cands) < n_candidates and attempts < n_candidates * 4:
        attempts += 1
        shots = rng.sample(pool_rows, min(5, len(pool_rows)))
        demo = "\n".join(
            f"Input: {r['text'][:160]}\nOutput: {lab[r['label']]}" for r in shots)
        meta = (
            "I gave a friend an instruction and five examples. Read the "
            "input/output examples below and write the single-sentence "
            "instruction that maps each input to its output. The output is "
            f"always exactly one of: {', '.join(ds.label_words())}.\n\n"
            f"{demo}\n\nThe instruction was:")
        ids = executor._encode(meta)
        with torch.no_grad():
            gen = lm.generate(ids, max_new_tokens=40, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        text = text.strip().strip('"').split("\n")[0].strip()
        if text and text.lower() not in seen and len(text) > 8:
            seen.add(text.lower())
            cands.append(text)
    return cands


# ---------------------------------------------------------------------------
# FIXED candidate pool (for ape-candidate-scoring / ape-search-strategy)
# ---------------------------------------------------------------------------
# A FIXED, DETERMINISTIC, hand-curated pool of candidate instructions spanning
# clearly-helpful task descriptions down to deliberately-misleading distractors.
# The pool is identical across runs and baselines (order fixed) so the scoring /
# strategy tasks reduce to "how well does your ESTIMATOR / SEARCH surface the
# candidate that truly generalizes to the held-out TEST set". Under the CALIBRATED
# executor a genuine task description separates cleanly from a distractor, and dev
# accuracy correlates with test accuracy, so a good selector wins monotonically.
# LM-induced candidates from a 0.5B model are too noisy to anchor a metric, so the
# pool is curated (induce_instructions remains available to the SEARCH proposer).
# NOTE the ORDER is fixed and deliberately NOT sorted by quality: index 0 is a
# DISTRACTOR, so a degenerate selector (constant estimator -> argmax picks index 0,
# or "pick candidates[0]") lands on a poor instruction and scores near the class
# prior, while a discriminating estimator / search must actively surface the genuine
# task descriptions to win. Good and bad instructions are interleaved so no positional
# shortcut helps. (Curated because 0.5B LM-induced candidates are too noisy to anchor.)
_CANDIDATE_POOL = {
    "topic": [
        "Respond with a random category regardless of the text.",            # distractor
        "Read the news article and identify which subject area it belongs to.",  # good
        "Say something about the text below.",                               # vague
        "Determine the general topic that best describes the news text below.",  # good
        "Answer the opposite of the correct category.",                      # distractor
        "Classify this news story into its news section by its main subject.",   # good
        "Summarize the passage in a single word.",                           # vague
        "Ignore the news and answer the same thing every time.",             # distractor
    ],
    "sentiment": [
        "Reply with a random sentiment ignoring the review.",                # distractor
        "Read the review and judge whether the opinion expressed is favorable or unfavorable.",  # good
        "Say something about the review below.",                             # vague
        "Determine the overall attitude of the review toward the film.",     # good
        "Always give the opposite of the reviewer's opinion.",               # distractor
        "Decide whether the reviewer liked or disliked the movie.",          # good
        "Describe the review in one word.",                                  # vague
        "Answer the same thing for every review.",                           # distractor
    ],
}


def build_candidate_pool(executor: "Executor", ds: "Dataset", seed: int = 42):
    """FIXED, DETERMINISTIC candidate-instruction pool for the scoring/strategy
    tasks (hand-curated; identical across runs and baselines). `executor`/`seed`
    are accepted for signature stability but the pool is deterministic and does not
    depend on them. Returns list[str]."""
    seen, out = set(), []
    for c in _CANDIDATE_POOL[ds.task]:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(c.strip())
    return out


# ---------------------------------------------------------------------------
# Scoring helpers (FIXED)
# ---------------------------------------------------------------------------
def accuracy(preds, rows) -> float:
    correct = sum(int(p == r["label"]) for p, r in zip(preds, rows))
    return correct / max(1, len(rows))


def majority_prior_accuracy(rows) -> float:
    """Accuracy of the degenerate always-majority-class predictor — the FLOOR any
    real instruction must beat."""
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    return max(c.values()) / len(rows)


# ---------------------------------------------------------------------------
# Shared engine for the EXPANDED ape-* research questions (appended; additive —
# does not change the fixed executor or the three original tasks). These helpers
# let each new editable surface plug a distinct function (proposer, meta-prompt,
# exemplar selector, paraphraser, calibration-input chooser, budgeted search) into
# the SAME proven frozen-LM / calibrated-forced-choice / disjoint pool-dev-test
# pipeline. Every expanded harness prints the SAME canonical APE_* lines so a
# single parser pattern covers all of them.
# ---------------------------------------------------------------------------
def lm_continuation(executor: "Executor", user_msg: str, max_new_tokens: int = 40) -> str:
    """Deterministic greedy continuation of the frozen LM for a fully-assembled
    USER message (chat template applied). Shared fixed decode for all induction /
    paraphrase surfaces. Returns the stripped generated text (first line)."""
    import torch
    ids = executor._encode(user_msg)
    with torch.no_grad():
        gen = executor.lm.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=executor.tok.eos_token_id)
    text = executor.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    return text.strip().strip('"').split("\n")[0].strip()


def induce_from_exemplars(executor: "Executor", exemplars, n_candidates: int,
                          meta_prompt: str | None = None, seed: int = 42,
                          counter: dict | None = None) -> list[str]:
    """Reverse-mode induction with AGENT-CHOSEN exemplars and an OPTIONAL agent
    meta-prompt. `exemplars` is the (surface-selected) labeled pool rows shown to
    the LM; `meta_prompt`, if given, is a format string with {demo} and {labels}
    that replaces the default reverse-mode template (ape-meta-prompt surface). Each
    LM.generate call increments counter["gen"] when a counter dict is supplied (for
    the shared-budget surface). Deterministic across seeds."""
    ds = executor.ds
    lab = ds.labels
    labels_str = ", ".join(ds.label_words())
    demo = "\n".join(
        f"Input: {r['text'][:160]}\nOutput: {lab[r['label']]}" for r in exemplars)
    if meta_prompt:
        tmpl = meta_prompt
        prompt = (tmpl.replace("{demo}", demo).replace("{labels}", labels_str)
                  if ("{demo}" in tmpl or "{labels}" in tmpl)
                  else tmpl + "\n\n" + demo + "\n\nThe instruction was:")
    else:
        prompt = (
            "I gave a friend an instruction and five examples. Read the "
            "input/output examples below and write the single-sentence "
            "instruction that maps each input to its output. The output is "
            f"always exactly one of: {labels_str}.\n\n{demo}\n\nThe instruction was:")
    cands: list[str] = []
    seen = set()
    text = lm_continuation(executor, prompt)
    if counter is not None:
        counter["gen"] = counter.get("gen", 0) + 1
    if text and text.lower() not in seen and len(text) > 8:
        seen.add(text.lower()); cands.append(text)
    # A single deterministic decode yields one candidate per distinct exemplar set;
    # the surface controls diversity via its exemplar choice / meta-prompt. Return
    # what we have (the harness pads nothing — selection runs over whatever the
    # surface produced plus its own anchors).
    return cands


def paraphrase_instruction(executor: "Executor", seed_instruction: str,
                           n_variants: int = 4, seed: int = 42) -> list[str]:
    """Ask the frozen LM to REWRITE `seed_instruction` into meaning-preserving
    paraphrases (instruction-paraphrase vs from-scratch surface). Deterministic
    greedy decode over a few distinct paraphrase prompts; returns distinct
    rewrites (excluding the seed itself)."""
    rng = random.Random(seed)
    frames = [
        "Rewrite the following instruction in different words, keeping the exact same meaning. Output only the rewritten instruction.\nInstruction: {s}\nRewritten:",
        "Paraphrase this task instruction concisely, preserving its intent. Output only the paraphrase.\nInstruction: {s}\nParaphrase:",
        "State the same classification instruction another way, in one sentence.\nInstruction: {s}\nAlternative:",
        "Rephrase the instruction below without changing what it asks the model to do.\nInstruction: {s}\nRephrase:",
    ]
    rng.shuffle(frames)
    out, seen = [], {seed_instruction.strip().lower()}
    for f in frames[:max(1, n_variants)]:
        t = lm_continuation(executor, f.format(s=seed_instruction.strip()))
        k = t.lower()
        if t and len(t) > 6 and k not in seen:
            seen.add(k); out.append(t)
    return out


def calibrated_dev_accuracy(executor: "Executor", instruction: str, dev_rows,
                            cal_inputs) -> float:
    """Dev accuracy of `instruction` under an AGENT-CHOSEN calibration: each
    input's raw per-label log-prob is debiased by the MEAN content-free label
    log-prob over `cal_inputs` (Calibrate-Before-Use, Zhao et al. 2021) rather than
    the FIXED "N/A" used for the held-out test metric. Selection-only: the test
    score still uses the fixed "N/A" calibration in Executor.predict. A poor
    calibration (e.g. a label-leaking string) fails to debias and mis-ranks
    candidates; a good set of content-free inputs (e.g. "N/A", "", "the") gives a
    stable selection signal."""
    import numpy as np
    if not cal_inputs:
        cal_inputs = ["N/A"]
    # mean content-free label log-prob over the agent's calibration inputs
    cal = np.zeros(executor.ds.n_class, dtype=np.float64)
    for ci in cal_inputs:
        cal += executor._raw_label_logprobs(
            executor._build_user_msg(instruction, str(ci)))
    cal /= float(len(cal_inputs))
    raw_rows = executor._raw_label_logprobs_many([
        executor._build_user_msg(instruction, row["text"])
        for row in dev_rows
    ])
    executor.n_exec_calls += len(dev_rows)
    preds = [int(np.argmax(raw - cal)) for raw in raw_rows]
    return float(np.mean([int(p == r["label"]) for p, r in zip(preds, dev_rows)]))


def select_best_by_dev(executor: "Executor", ds: "Dataset", candidates):
    """Fixed APE selection: highest dev execution-accuracy. Returns (instr, acc)."""
    best, best_acc = candidates[0], -1.0
    for c in candidates:
        a = executor.dev_accuracy(c, ds.dev)
        if a > best_acc:
            best_acc, best = a, c
    return best, best_acc


def evaluate_selected_test(
    executor: "Executor", ds: "Dataset", instruction: str
) -> tuple[float, int]:
    """Evaluate exactly one dev-selected instruction on the official split."""
    if executor.selected_test_passes != 0:
        raise RuntimeError("the evaluation split may be traversed exactly once")
    executor.selected_test_passes = 1
    preds = executor.predict(instruction, ds.test)
    correct = sum(int(pred == row["label"]) for pred, row in zip(preds, ds.test))
    executor.last_test_correct = correct
    return correct / len(ds.test), correct


def eval_test(executor: "Executor", ds: "Dataset", instruction: str) -> float:
    value, _correct = evaluate_selected_test(executor, ds, instruction)
    return value


def emit_selected_metrics(
    executor: "Executor",
    ds: "Dataset",
    instruction: str,
    t0: float,
    *,
    n_candidates: int,
    dev_exec_calls: int,
) -> None:
    """Evaluate the selected instruction and emit one terminal completion proof."""
    if not isinstance(instruction, str):
        raise TypeError("selected instruction must be a string")
    if n_candidates < 1 or dev_exec_calls < 0:
        raise ValueError("invalid candidate or dev-execution inventory")
    test_acc, correct = evaluate_selected_test(executor, ds, instruction)
    elapsed = time.time() - t0
    selected_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    floor = majority_prior_accuracy(ds.test)
    shown = instruction.replace("\n", " ")[:200]
    print(
        f"APE_FLOOR majority_prior_acc={floor:.9f} dataset={ds.name}",
        flush=True,
    )
    print(f'APE_CHOSEN instruction="{shown}"', flush=True)
    print(
        f"APE_RESULT status=complete protocol={PROTOCOL} dataset={ds.name} "
        f"eval_split={ds.evaluation_split} pool_n={len(ds.pool)} dev_n={len(ds.dev)} "
        f"eval_n={len(ds.test)} correct={correct} accuracy={test_acc:.9f} "
        f"selected_test_passes={executor.selected_test_passes} "
        f"n_candidates={n_candidates} dev_exec_calls={dev_exec_calls} "
        f"selected_sha256={selected_sha} data_sha256={ds.data_sha256} "
        f"model_sha256={ds.model_sha256} protocol_sha256={ds.protocol_sha256} "
        f"elapsed={elapsed:.6f}",
        flush=True,
    )


def emit_pool_metrics(executor: "Executor", ds: "Dataset", candidates,
                      chosen: str, t0: float, extra: str = "") -> None:
    """Canonical completion path for pool-based surfaces.

    The evaluation split is never used to compute an oracle over the candidate pool;
    it is traversed only for the single instruction selected on train/dev data.
    """
    del extra
    emit_selected_metrics(
        executor,
        ds,
        chosen,
        t0,
        n_candidates=len(candidates),
        dev_exec_calls=executor.n_exec_calls,
    )
