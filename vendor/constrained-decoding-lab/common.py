"""Shared utilities for the constrained-decoding (cd-*) MLS-Bench tasks.

Fixed, un-gameable evaluation path for INFERENCE-TIME structured decoding:

  * A single FROZEN small instruction LM (Qwen2.5-0.5B-Instruct) is loaded once.
  * A FIXED set of structured-output prompts (GSM8K numeric answers, or a
    forced-choice classification set) is decoded.
  * The agent controls ONLY the *decoding policy*: how to turn the task's answer
    space into a constraint (regex / choice set / FSM) and WHAT part of the
    generation to constrain (full schema vs only the final answer field, with an
    optional free-form reasoning preamble).
  * The model weights, prompts, gold answers, token budget, sampling temperature
    (greedy), and the scoring extractor are ALL fixed here.

The score jointly measures STRUCTURAL VALIDITY and TASK CORRECTNESS. Because a
sample is only "correct" if it is BOTH structurally valid AND matches the gold
answer, a decoder that always emits a valid-but-wrong answer cannot beat a
decoder that actually gets the answer right — validity alone never moves the
score. See harness_*.py for the emitted metric lines.

Datasets and the model are staged offline under $CD_DATA / $CD_MODEL so the
container runs with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Callable


MODEL_PARAMETERS = 494_032_768

# ---------------------------------------------------------------------------
# Paths / model
# ---------------------------------------------------------------------------

def model_path() -> str:
    """FROZEN small instruction LM. All cd-* tasks decode from this model."""
    return os.environ.get(
        "CD_MODEL", "/data/constrained-decoding/models/Qwen2.5-0.5B-Instruct"
    )


def data_root() -> Path:
    return Path(os.environ.get("CD_DATA", "/data/constrained-decoding/data"))


def set_seeds(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_SURFACE_SOURCE_BYTES = 64 * 1024
_SURFACE_AST_NODES = 512
_GSM8K_TEST_COUNT = 1319
_AGNEWS_TEST_COUNT = 7600
_AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
_DECODER_FIELDS = {
    "prompt",
    "answer_regex",
    "choices",
    "choice_labels",
    "preamble_regex",
    "trigger",
    "max_answer_tokens",
    "max_free_tokens",
}


def _surface_error(message: str) -> ValueError:
    return ValueError(f"unsafe constrained-decoding surface: {message}")


def _eval_surface_expr(node: ast.AST, values: dict[str, object]) -> object:
    """Evaluate the tiny expression language allowed inside build_decoder.

    This deliberately does not use Python ``eval``. The editable file may build
    strings, copy the fixed label list, and index that list; it cannot execute a
    callable, import a module, inspect the verifier, print output, or terminate
    the process.
    """
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, int, float, bool)):
            return node.value
        raise _surface_error(f"constant type {type(node.value).__name__} is not allowed")
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise _surface_error(f"unknown name {node.id!r}")
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_eval_surface_expr(item, values) for item in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
            elif isinstance(item, ast.FormattedValue) and item.conversion == -1:
                parts.append(str(_eval_surface_expr(item.value, values)))
            else:
                raise _surface_error("f-strings may only interpolate approved values")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_surface_expr(node.left, values)
        right = _eval_surface_expr(node.right, values)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise _surface_error("only string concatenation is allowed")
    if isinstance(node, ast.Call):
        if node.keywords:
            raise _surface_error("helper calls may not use keyword arguments")
        if (isinstance(node.func, ast.Name) and node.func.id == "list"
                and len(node.args) == 1):
            value = _eval_surface_expr(node.args[0], values)
            if not isinstance(value, (list, tuple)):
                raise _surface_error("list() input must be the fixed label sequence")
            return list(value)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "join"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)
                and len(node.args) == 1):
            value = _eval_surface_expr(node.args[0], values)
            if not isinstance(value, (list, tuple)) or not all(
                    isinstance(item, str) for item in value):
                raise _surface_error("str.join input must be the fixed label sequence")
            return node.func.value.value.join(value)
        raise _surface_error("function calls are not allowed")
    if isinstance(node, ast.Subscript):
        value = _eval_surface_expr(node.value, values)
        index = _eval_surface_expr(node.slice, values)
        if not isinstance(value, (list, tuple)) or not isinstance(index, int):
            raise _surface_error("only constant indexing of the fixed label sequence is allowed")
        try:
            return value[index]
        except IndexError as exc:
            raise _surface_error("label index is out of range") from exc
    raise _surface_error(f"expression {type(node).__name__} is not allowed")


def _parse_surface_function(sol_path: Path, attr: str) -> ast.FunctionDef:
    if attr != "build_decoder":
        raise _surface_error(f"unsupported surface name {attr!r}")
    if not sol_path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {sol_path}")
    source = sol_path.read_text()
    if len(source.encode()) > _SURFACE_SOURCE_BYTES:
        raise _surface_error("source exceeds 64 KiB")
    try:
        tree = ast.parse(source, filename=str(sol_path))
    except SyntaxError as exc:
        raise _surface_error(f"source does not parse: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _SURFACE_AST_NODES:
        raise _surface_error("AST exceeds 512 nodes")

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == attr
    ]
    if len(functions) != 1:
        raise _surface_error(f"solution must define exactly one {attr}")
    function = functions[0]
    if (function.decorator_list or function.args.posonlyargs
            or function.args.vararg is not None or function.args.kwarg is not None
            or function.args.kwonlyargs or function.args.defaults
            or function.args.kw_defaults):
        raise _surface_error("build_decoder must be undecorated with fixed positional arguments")
    argument_names = [arg.arg for arg in function.args.args]
    if argument_names not in (["question", "tok"], ["text", "labels", "tok"]):
        raise _surface_error(
            "build_decoder signature must be (question, tok) or (text, labels, tok)"
        )

    for node in tree.body:
        if node is function:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["common"]:
            continue
        raise _surface_error("top-level executable statements and imports are forbidden")
    return function


def load_surface(sol_path: str, attr: str):
    """Build a decoder from a restricted AST without executing agent Python."""
    function = _parse_surface_function(Path(sol_path).resolve(), attr)
    argument_names = [arg.arg for arg in function.args.args]

    def build_decoder(*args):
        if len(args) != len(argument_names):
            raise TypeError(f"{attr} expects {len(argument_names)} arguments")
        values: dict[str, object] = dict(zip(argument_names, args))
        returned: DecodeSpec | None = None
        for statement in function.body:
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise _surface_error("assignments must target one local name")
                name = statement.targets[0].id
                if name in argument_names or name.startswith("__"):
                    raise _surface_error(f"invalid assignment target {name!r}")
                values[name] = _eval_surface_expr(statement.value, values)
                continue
            if isinstance(statement, ast.Return):
                call = statement.value
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "common"
                        and call.func.attr == "DecodeSpec"):
                    raise _surface_error("final return must construct common.DecodeSpec")
                if len(call.args) > 1:
                    raise _surface_error("DecodeSpec accepts at most one positional prompt")
                kwargs: dict[str, object] = {}
                if call.args:
                    kwargs["prompt"] = _eval_surface_expr(call.args[0], values)
                for keyword in call.keywords:
                    if keyword.arg is None or keyword.arg not in _DECODER_FIELDS:
                        raise _surface_error(f"unknown DecodeSpec field {keyword.arg!r}")
                    if keyword.arg in kwargs:
                        raise _surface_error(f"duplicate DecodeSpec field {keyword.arg!r}")
                    kwargs[keyword.arg] = _eval_surface_expr(keyword.value, values)
                returned = DecodeSpec(**kwargs)
                continue
            raise _surface_error(f"statement {type(statement).__name__} is not allowed")
        if returned is None or not isinstance(function.body[-1], ast.Return):
            raise _surface_error("build_decoder must end with one DecodeSpec return")
        return returned

    return build_decoder


_TOKENIZER = None
_MODEL = None


def load_model():
    """Load (once) the frozen model + tokenizer onto GPU. Greedy, fp16."""
    global _TOKENIZER, _MODEL
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = model_path()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, local_files_only=True
    ).eval()
    if not torch.cuda.is_available():
        raise RuntimeError("constrained-decoding verification requires CUDA")
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != MODEL_PARAMETERS:
        raise RuntimeError(
            f"unexpected frozen model size: {actual_parameters} != {MODEL_PARAMETERS}"
        )
    model = model.cuda()
    _TOKENIZER, _MODEL = tok, model
    return tok, model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_gsm8k(n: int | None = None) -> list[dict]:
    """Pinned GSM8K numeric-answer test split.

    Each item: {"question": str, "gold": str}  (gold = the integer answer as a
    string; GSM8K answers are integers).
    """
    if n not in (None, _GSM8K_TEST_COUNT):
        raise ValueError(
            f"GSM8K verifier requires the full {_GSM8K_TEST_COUNT}-example test split"
        )
    path = data_root() / "gsm8k.json"
    payload = path.read_bytes()
    expected_sha256 = "02ed017f9052a9e70777d01f388ba30153d04ccf5d06503e4d76a86005d8114e"
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("GSM8K verifier data digest mismatch")
    items = json.loads(payload)
    if (not isinstance(items, list) or len(items) != _GSM8K_TEST_COUNT
            or any(not isinstance(item, dict)
                   or set(item) != {"question", "gold"}
                   or not isinstance(item["question"], str)
                   or not item["question"].strip()
                   or not isinstance(item["gold"], str)
                   or re.fullmatch(r"-?\d+", item["gold"]) is None
                   for item in items)):
        raise ValueError("GSM8K verifier data is incomplete or malformed")
    return items


def load_classification(n: int | None = None) -> tuple[list[dict], list[str]]:
    """Full fixed AG News forced-choice test split.

    Returns (items, labels) where each item is {"text": str, "gold": <label>}
    and `labels` is the fixed ordered label set the answer must come from.
    """
    if n not in (None, _AGNEWS_TEST_COUNT):
        raise ValueError(
            f"AG News verifier requires the full {_AGNEWS_TEST_COUNT}-example test split"
        )
    path = data_root() / "classification.json"
    payload = path.read_bytes()
    expected_sha256 = "33645f5c37148a6b05003c8fbbd8994b1c863a7642e9f1a52fda70dff6aa8a4e"
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("AG News verifier data digest mismatch")
    blob = json.loads(payload)
    if not isinstance(blob, dict) or set(blob) != {"labels", "items"}:
        raise ValueError("AG News verifier data is malformed")
    labels = blob["labels"]
    items = blob["items"]
    if (labels != _AGNEWS_LABELS or not isinstance(items, list)
            or len(items) != _AGNEWS_TEST_COUNT
            or any(not isinstance(item, dict)
                   or set(item) != {"text", "gold"}
                   or not isinstance(item["text"], str)
                   or not item["text"].strip()
                   or item["gold"] not in _AGNEWS_LABELS
                   for item in items)):
        raise ValueError("AG News verifier data is incomplete or malformed")
    return items, labels


# ---------------------------------------------------------------------------
# Answer extraction (FIXED — not agent-controlled)
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"-?\d[\d,]*")


def extract_gsm8k_answer(final_answer_text: str) -> str | None:
    """FIXED extractor for the numeric task.

    Given the string the decoder committed to for the ANSWER FIELD, return the
    canonical integer string, or None if no integer is present. Commas are
    stripped. This is deliberately lenient about surrounding characters so that
    both a bare "42" and a JSON '{"answer": 42}' answer-field value parse — the
    decoder's job is to *emit* a parseable answer, and correctness is judged on
    the integer value, not on the surrounding syntax.
    """
    m = _INT_RE.search(final_answer_text or "")
    if not m:
        return None
    return m.group(0).replace(",", "")


def normalize_gold_int(gold: str) -> str:
    return str(gold).replace(",", "").strip()


# ---------------------------------------------------------------------------
# Constrained-generation engine (FIXED harness machinery)
# ---------------------------------------------------------------------------

class DecodeSpec:
    """What the agent's decoder returns for a single prompt.

    Attributes
    ----------
    prompt : str
        The full prompt string fed to the model (agent controls the instruction
        text / schema wording).
    preamble_regex : str | None
        Optional regex describing a FREE-FORM reasoning region emitted *before*
        the answer is constrained. If None, no free reasoning region — the
        constraint applies from the first generated token. If given, the model
        generates freely (subject only to this loose regex, typically
        ``".*"``-like) until the `trigger` string appears, at which point the
        tight `answer_regex` / `choices` constraint kicks in. This is how a
        decoder implements "reason first, then constrain only the answer".
    trigger : str | None
        A literal string that, once emitted during the free region, switches the
        decoder into the constrained (answer) region. Required iff a free region
        is used.
    answer_regex : str | None
        Regex the ANSWER region must match (mutually exclusive with `choices`).
    choices : list[str] | None
        A fixed set of literal strings the answer must be exactly one of
        (forced-choice). Mutually exclusive with `answer_regex`.
    max_answer_tokens : int
        Cap on tokens generated inside the answer region.
    max_free_tokens : int
        Budget for the FREE reasoning region (how many tokens the model may
        reason for before the `trigger` must appear). This is itself a design
        axis: too small starves reasoning (accuracy drops toward the
        answer-only case), too large wastes compute. Ignored when there is no
        free region.
    """

    def __init__(
        self,
        prompt: str,
        answer_regex: str | None = None,
        choices: list[str] | None = None,
        choice_labels: list[str] | None = None,
        preamble_regex: str | None = None,
        trigger: str | None = None,
        max_answer_tokens: int = 16,
        max_free_tokens: int = 320,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            print("SURFACE_ERROR constrained-decoding prompt must be non-empty", flush=True)
            raise TypeError("prompt must be a non-empty string")
        if (answer_regex is None) == (choices is None):
            raise ValueError("Provide exactly one of answer_regex / choices")
        if (preamble_regex is not None) != (trigger is not None):
            raise ValueError("preamble_regex and trigger must be given together")
        if answer_regex is not None and (not isinstance(answer_regex, str) or not answer_regex):
            print("SURFACE_ERROR constrained-decoding answer_regex must be non-empty",
                  flush=True)
            raise TypeError("answer_regex must be a non-empty string")
        if choices is not None:
            if (not isinstance(choices, list) or not choices or
                    any(not isinstance(choice, str) or not choice for choice in choices) or
                    len(set(choices)) != len(choices)):
                print("SURFACE_ERROR constrained-decoding choices must be unique strings",
                      flush=True)
                raise TypeError("choices must be a non-empty list of unique strings")
        if choice_labels is not None:
            if (choices is None or not isinstance(choice_labels, list)
                    or len(choice_labels) != len(choices)
                    or any(not isinstance(label, str) or not label
                           for label in choice_labels)
                    or len(set(choice_labels)) != len(choice_labels)):
                raise TypeError(
                    "choice_labels must contain one unique non-empty label per choice"
                )
        if preamble_regex is not None and not isinstance(preamble_regex, str):
            print("SURFACE_ERROR constrained-decoding preamble_regex must be a string",
                  flush=True)
            raise TypeError("preamble_regex must be a string")
        if preamble_regex is not None and preamble_regex != r"[\s\S]*":
            raise ValueError("preamble_regex must describe an unrestricted reasoning region")
        if trigger is not None and (not isinstance(trigger, str) or not trigger):
            print("SURFACE_ERROR constrained-decoding trigger must be non-empty", flush=True)
            raise TypeError("trigger must be a non-empty string")
        for name, value, limit in (
            ("max_answer_tokens", max_answer_tokens, 256),
            ("max_free_tokens", max_free_tokens, 4096),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
                print(f"SURFACE_ERROR constrained-decoding {name}={value!r}", flush=True)
                raise ValueError(f"invalid {name}")
        self.prompt = prompt
        self.answer_regex = answer_regex
        self.choices = choices
        self.choice_labels = choice_labels
        self.preamble_regex = preamble_regex
        self.trigger = trigger
        self.max_answer_tokens = max_answer_tokens
        self.max_free_tokens = max_free_tokens


def _regex_to_fsm(pattern: str):
    """Compile a regex into an `interegular` FSM (character-level DFA)."""
    import interegular

    return interegular.parse_pattern(pattern).to_fsm()


class _CharFSMMask:
    """Character-level FSM that yields, for the current state, the set of
    ALLOWED next *characters*, and whether the current state is accepting.

    We drive it with the *decoded string so far in the answer region*. At each
    generation step we re-derive, from the FSM state reached by the committed
    answer string, which tokens are legal continuations by checking, for each
    candidate token's string, whether feeding its characters keeps the FSM alive
    (does not hit the dead state). This is the standard char-DFA → token mask
    lift; it is O(vocab) per step but fine for short answer regions on a small
    model and a few hundred prompts.
    """

    DEAD = object()  # sentinel for the dead state

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.fsm = _regex_to_fsm(pattern)
        self._anything = interegular_anything()
        # Concrete characters the pattern ever cares about (everything else maps
        # to `anything_else`). Used to pre-filter the vocab cheaply.
        self.concrete_chars = frozenset(
            k for k in self.fsm.alphabet.keys() if isinstance(k, str)
        )

    @property
    def initial(self):
        return self.fsm.initial

    def step(self, state, ch):
        """Advance one character from `state`. Returns next state or DEAD."""
        if state is self.DEAD:
            return self.DEAD
        fsm = self.fsm
        sym = fsm.alphabet.get(ch, fsm.alphabet.get(self._anything, None))
        if sym is None:
            return self.DEAD
        nxt = fsm.map.get(state, {}).get(sym, None)
        return self.DEAD if nxt is None else nxt

    def run_from(self, state, s: str):
        """Advance the string `s` from `state`. Returns end state or DEAD."""
        for ch in s:
            state = self.step(state, ch)
            if state is self.DEAD:
                return self.DEAD
        return state

    def is_final(self, state) -> bool:
        return state is not self.DEAD and state in self.fsm.finals

    # string-based convenience (used by choice-free paths / tests)
    def _run(self, s: str):
        state = self.run_from(self.fsm.initial, s)
        return state, (state is not self.DEAD)

    def accepts(self, s: str) -> bool:
        state, alive = self._run(s)
        return alive and state in self.fsm.finals

    def can_extend(self, s: str) -> bool:
        _, alive = self._run(s)
        return alive


def interegular_anything():
    import interegular
    return interegular.fsm.anything_else


_REGEX_CANDIDATE_CACHE: dict[tuple[int, str], list[tuple[int, str]]] = {}


def _candidate_tokens(mask: "_CharFSMMask", vocab_strings: dict[int, str]):
    """Return every token that can advance at least one state in this FSM.

    Tight numeric/label expressions never transition on ``anything_else`` and
    retain the cheap concrete-character filter. General expressions such as
    ``[^\\n]+`` do use that transition, so filtering them to concrete alphabet
    characters would incorrectly remove every normal token. Those expressions
    use a complete state-reachability scan, cached once per tokenizer/pattern.
    """
    cache_key = (id(vocab_strings), mask.pattern)
    cached = _REGEX_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    chars = mask.concrete_chars
    cands: list[tuple[int, str]] = []
    anything_symbol = mask.fsm.alphabet.get(mask._anything)
    accepts_anything = anything_symbol is not None and any(
        anything_symbol in transitions for transitions in mask.fsm.map.values()
    )
    if not accepts_anything:
        for tid, s in vocab_strings.items():
            if s and all(c in chars for c in s):
                cands.append((tid, s))
    else:
        states = [mask.initial]
        states.extend(state for state in mask.fsm.states if state != mask.initial)
        for tid, s in vocab_strings.items():
            if s and any(mask.run_from(state, s) is not mask.DEAD for state in states):
                cands.append((tid, s))
    _REGEX_CANDIDATE_CACHE[cache_key] = cands
    return cands


def _allowed_from_state(
    mask: "_CharFSMMask",
    state,
    cands: list[tuple[int, str]],
    eos_ok: bool,
    eos_id: int,
) -> tuple[set[int], dict[int, object]]:
    """Given the current FSM `state`, return (allowed_token_ids, next_state_map)
    over the prefiltered candidate tokens. EOS is allowed iff the state is
    already accepting (so the answer can terminate at a valid boundary)."""
    allowed: set[int] = set()
    nxt_state: dict[int, object] = {}
    for tid, s in cands:
        if tid == eos_id:
            continue
        end = mask.run_from(state, s)
        if end is not mask.DEAD:
            allowed.add(tid)
            nxt_state[tid] = end
    if eos_ok and mask.is_final(state):
        allowed.add(eos_id)
    return allowed, nxt_state


def _build_vocab_strings(tok) -> dict[int, str]:
    """Map token id -> its decoded surface string (single-token decode).

    Cached on the tokenizer object.
    """
    cached = getattr(tok, "_cd_vocab_strings", None)
    if cached is not None:
        return cached
    vocab = tok.get_vocab()
    special_ids = set(getattr(tok, "all_special_ids", ()))
    out: dict[int, str] = {}
    for _tokstr, tid in vocab.items():
        if tid in special_ids:
            continue
        try:
            s = tok.convert_tokens_to_string([_tokstr])
        except Exception as exc:
            print(f"TOKEN_SURFACE_ERROR token_id={tid} token={_tokstr!r}: {exc}", flush=True)
            raise RuntimeError(
                f"failed to convert tokenizer vocabulary entry {tid} ({_tokstr!r})"
            ) from exc
        out[tid] = s
    tok._cd_vocab_strings = out  # type: ignore[attr-defined]
    return out


def _encode_prompt(tok, user_prompt: str):
    """Wrap the agent's prompt in the FIXED chat template and encode it."""
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    except Exception as exc:
        print(f"PROMPT_TEMPLATE_ERROR constrained-decoding: {exc}", flush=True)
        raise RuntimeError("failed to apply the fixed constrained-decoding chat template") from exc


def run_decode(spec: DecodeSpec, seed: int = 42) -> dict:
    """Execute one constrained decode following `spec`. FIXED machinery.

    Returns a dict with:
      answer_text : str   the text committed inside the ANSWER region
      full_text   : str   everything the model generated (reasoning + answer)
      valid       : bool   True iff the answer region satisfied the constraint
                           (FSM accepted / a full choice was emitted)
    """
    import torch

    set_seeds(seed)
    tok, model = load_model()
    device = next(model.parameters()).device
    eos_id = tok.eos_token_id
    vocab_strings = _build_vocab_strings(tok)

    # The agent-provided `prompt` is the USER message; wrap it in the model's
    # chat template (fixed) so a small instruction LM behaves as intended. The
    # generation prompt is appended so the model starts the assistant turn.
    prompt_ids = _encode_prompt(tok, spec.prompt)
    input_ids = prompt_ids.to(device)
    generated: list[int] = []
    full_chunks: list[str] = []

    # -- Phase 1: optional FREE reasoning region until the trigger appears -----
    in_answer = True
    if spec.preamble_regex is not None:
        in_answer = False
        free_text = ""
        max_free = spec.max_free_tokens
        with torch.no_grad():
            for _ in range(max_free):
                out = model(torch.cat([input_ids, _t(generated, device)], dim=1))
                logits = out.logits[0, -1]
                if not torch.isfinite(logits).all():
                    print("CD_NONFINITE free_generation_logits", flush=True)
                    raise FloatingPointError("non-finite constrained-decoding logits")
                nxt = int(torch.argmax(logits).item())
                if nxt == eos_id:
                    break
                generated.append(nxt)
                free_text += vocab_strings.get(nxt, "")
                if spec.trigger and spec.trigger in free_text:
                    in_answer = True
                    break
        full_chunks.append(free_text)

    if not in_answer:
        return {
            "answer_text": "",
            "full_text": "".join(full_chunks),
            "valid": False,
        }

    # -- Phase 2: CONSTRAINED answer region -----------------------------------
    answer_text = ""
    valid = False
    if spec.choices is not None:
        # Forced choice: pick the choice with the highest summed logprob under
        # greedy teacher-forcing over the fixed choice set (this is the correct,
        # boundary-safe way to constrain to a small literal label set — no
        # tokenization drift).
        answer_text, valid = _score_choices(
            tok, model, device, input_ids, generated, spec.choices
        )
        if spec.choice_labels is not None:
            mapped_answer = spec.choice_labels[spec.choices.index(answer_text)]
        else:
            mapped_answer = None
        full_chunks.append(answer_text)
    else:
        mask = _CharFSMMask(spec.answer_regex)  # type: ignore[arg-type]
        cands = _candidate_tokens(mask, vocab_strings)
        state = mask.initial
        with torch.no_grad():
            for step in range(spec.max_answer_tokens):
                eos_ok = step > 0  # don't allow an empty answer
                allowed, nxt_state = _allowed_from_state(
                    mask, state, cands, eos_ok, eos_id
                )
                if not allowed:
                    break
                out = model(torch.cat([input_ids, _t(generated, device)], dim=1))
                logits = out.logits[0, -1]
                if not torch.isfinite(logits).all():
                    print("CD_NONFINITE constrained_generation_logits", flush=True)
                    raise FloatingPointError("non-finite constrained-decoding logits")
                neg = torch.full_like(logits, float("-inf"))
                idx = torch.tensor(sorted(allowed), device=device)
                neg[idx] = logits[idx]
                nxt = int(torch.argmax(neg).item())
                if nxt == eos_id:
                    break
                generated.append(nxt)
                answer_text += vocab_strings.get(nxt, "")
                state = nxt_state[nxt]
        valid = mask.is_final(state)

    result = {
        "answer_text": answer_text,
        "full_text": "".join(full_chunks),
        "valid": bool(valid),
    }
    if spec.choices is not None and mapped_answer is not None:
        result["mapped_answer"] = mapped_answer
    return result


def _t(ids: list[int], device):
    import torch
    if not ids:
        return torch.zeros((1, 0), dtype=torch.long, device=device)
    return torch.tensor([ids], dtype=torch.long, device=device)


def _score_choices(tok, model, device, input_ids, prefix_ids, choices):
    """Return (best_choice, True). Picks argmax summed logprob over the fixed
    choice set — a validity-guaranteed forced-choice decode."""
    import torch

    base = torch.cat([input_ids, _t(prefix_ids, device)], dim=1)
    best_choice, best_lp = choices[0], float("-inf")
    with torch.no_grad():
        for ch in choices:
            ch_ids = tok(ch, add_special_tokens=False).input_ids
            if not ch_ids:
                continue
            seq = torch.cat([base, _t(ch_ids, device)], dim=1)
            out = model(seq)
            logits = out.logits[0]
            if not torch.isfinite(logits).all():
                print("CD_NONFINITE choice_logits", flush=True)
                raise FloatingPointError("non-finite constrained-decoding choice logits")
            # logprob of each choice token given its prefix
            lp = 0.0
            start = base.shape[1] - 1
            for k, tid in enumerate(ch_ids):
                step_logits = logits[start + k]
                logprobs = torch.log_softmax(step_logits, dim=-1)
                lp += float(logprobs[tid].item())
            lp /= max(len(ch_ids), 1)  # length-normalized
            if not __import__("math").isfinite(lp):
                print(f"CD_NONFINITE choice_logprob choice={ch!r}", flush=True)
                raise FloatingPointError("non-finite constrained-decoding choice score")
            if lp > best_lp:
                best_lp, best_choice = lp, ch
    return best_choice, True
