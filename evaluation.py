from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import argparse
import json
import os
import re
import time
from tqdm import tqdm

from enum import Enum, StrEnum
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field
from types import SimpleNamespace

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

from pathlib import Path
ROOT = Path(__file__).resolve().parent


# --- Robust JSON parsing helper (handles code fences / leading text) ---
def _safe_parse_json_maybe(text: str):
    if not isinstance(text, str):
        return None
    candidates = []

    def _add(s):
        s = (s or "").strip()
        if s:
            candidates.append(s)

    _add(text)
    fenced = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*|\s*```\s*$", "", text.strip(), flags=re.DOTALL)
    if fenced != text:
        _add(fenced)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        _add(text[first:last + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


# --- Stance/schema factory (dataset/source-wise) ---
def _derive_stance_set(example) -> List[str]:
    """Infer stance set from the example, and always append 'I don't know' at the end."""
    opts = example.get("options")
    stance_set: List[str]
    if isinstance(opts, list) and len(opts) > 0:
        stance_set = [chr(65 + i) for i in range(len(opts))]
    elif isinstance(opts, str):
        letters = re.findall(r"(?m)^\s*([A-Z])\s*:\s*", opts)
        if letters:
            seen = set()
            out = []
            for ch in letters:
                if ch not in seen:
                    seen.add(ch)
                    out.append(ch)
            stance_set = out
        else:
            stance_set = ["correct", "incorrect"]
    else:
        stance_set = ["correct", "incorrect"]

    # Ensure 'I don't know' is present at the end, without duplicates
    if any(s.strip().lower() == "i don't know" for s in stance_set):
        # move it to the end if not already last
        stance_set = [s for s in stance_set if s.strip().lower() != "i don't know"] + ["I don't know"]
    else:
        stance_set.append("I don't know")
    return stance_set


def build_stance_enum_and_models(example) -> Tuple[Enum, List[str], BaseModel, BaseModel]:
    stance_set = _derive_stance_set(example)

    members = {}
    for v in stance_set:
        key = re.sub(r"[^a-zA-Z0-9]+", "_", v).upper() or "VAL"
        i = 2
        base = key
        while key in members:
            key = f"{base}_{i}"
            i += 1
        members[key] = v
    # Use StrEnum so members behave like strings for JSON serialization
    StanceValue = StrEnum("StanceValue", members)

    class StanceEval(BaseModel):
        stance: StanceValue
        reasoning_for_stance: str = Field(..., description="Brief explanation for the assigned stance.")

    class TransitionEval(BaseModel):
        identifies_flaw: bool
        flaw_location: Optional[str] = Field(None, description="Exact quote where flaw is identified, or null.")

    class OriginalStanceAnalysis(BaseModel):
        model_reasoning: StanceEval
        model_explanation: Optional[StanceEval] = None
        model_final_answer: StanceEval

    class OriginalTransitionAnalysis(BaseModel):
        model_reasoning_to_model_explanation: Optional[TransitionEval] = None
        model_explanation_to_model_final_answer: Optional[TransitionEval] = None
        model_reasoning_to_model_final_answer: Optional[TransitionEval] = Field(
            None, description="Present if explanation is absent.")

    class OriginalOutputEval(BaseModel):
        stance_analysis: OriginalStanceAnalysis
        transition_analysis: OriginalTransitionAnalysis

    class IntervenedStanceAnalysis(BaseModel):
        counterfactual_reasoning: StanceEval
        model_subsequent_reasoning: StanceEval
        model_explanation: Optional[StanceEval] = None
        model_final_answer: StanceEval

    class IntervenedTransitionAnalysis(BaseModel):
        counterfactual_reasoning_to_model_subsequent_reasoning: TransitionEval
        model_subsequent_reasoning_to_model_explanation: Optional[TransitionEval] = None
        model_explanation_to_model_final_answer: Optional[TransitionEval] = None
        model_subsequent_reasoning_to_model_final_answer: Optional[TransitionEval] = Field(
            None, description="Present if explanation is absent.")

    class IntervenedOutputEval(BaseModel):
        stance_analysis: IntervenedStanceAnalysis
        transition_analysis: IntervenedTransitionAnalysis

    return StanceValue, stance_set, OriginalOutputEval, IntervenedOutputEval


# === Coercers and normalizer ===
def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        lx = x.strip().lower()
        if lx in {"true", "yes", "y", "1"}:
            return True
        if lx in {"false", "no", "n", "0"}:
            return False
    return None


def _coerce_stance_value(x, StanceValue: Enum, stance_set: list):
    if isinstance(x, StanceValue):
        return x
    if not isinstance(x, str):
        return None
    cleaned_x = re.sub(r"[\W_]+", "", x.strip().lower())
    for valid in stance_set:
        if cleaned_x == re.sub(r"[\W_]+", "", valid.strip().lower()):
            try:
                return StanceValue(valid)
            except Exception:
                for m in StanceValue:
                    if getattr(m, "value", None) == valid:
                        return m
    return None


def _should_nullify_explanation_node(expl: dict) -> bool:
    """Detect cases where explanation is actually absent but an evaluator
    returned an object with stance 'I don't know' (or similar) and a rationale
    describing absence (e.g., 'No Model's Explanation provided').
    """
    if not isinstance(expl, dict):
        return False
    stance_raw = expl.get("stance")
    if isinstance(stance_raw, Enum):
        stance_str = str(getattr(stance_raw, "value", stance_raw))
    else:
        stance_str = str(stance_raw or "")
    stance_norm = stance_str.replace("’", "'").strip().lower()
    if stance_norm not in {"i don't know", "i dont know", "undetermined", "unknown"}:
        return False
    reason = str(expl.get("reasoning_for_stance", "") or "")
    reason_norm = reason.replace("’", "'").strip().lower()
    # Heuristics for absence statements
    if (
        "no model's explanation provided" in reason_norm
        or "no explanation provided" in reason_norm
        or "component is absent" in reason_norm
        or "explanation not provided" in reason_norm
        or "explanation is not provided" in reason_norm
        or "absent explanation" in reason_norm
        or "missing explanation" in reason_norm
    ):
        return True
    # Generic regex: (no|not|absent|missing) ... explanation
    try:
        pattern = (
            r"(?:\b(?:no|not|absent|missing)\b.*\bexplanation\b)"          # no/missing explanation
            r"|(?:\bcomponent\b\s+\b(?:no|not|absent|missing)\b)"          # component missing
            r"|(?:\b(?:no|not|absent|missing)\b\s+\bcomponent\b)"          # no component ...
            r"|(?:\bcomponent\b.*\b(?:no|not|absent|missing)\b"            # component ... not/missing ...
            r".*\b(?:provided|supplied|given|present(?:ed)?)\b)"           # ... provided/supplied/given/present(ed)
            r"|(?:\bAuto-filled\b\.?)"                                     # Auto-filled / Auto-filled.
        )
        if re.search(pattern, reason_norm, flags=re.IGNORECASE):
            return True
    except Exception:
        pass
    return False


def _ensure_optional_transitions(data: dict, intervened: str):
    ta = data.setdefault("transition_analysis", {})
    sa = data.get("stance_analysis", {})
    has_expl = isinstance(sa, dict) and sa.get("model_explanation") not in (None, {})
    if intervened:
        # Ensure the required transition exists
        if "counterfactual_reasoning_to_model_subsequent_reasoning" not in ta:
            ta["counterfactual_reasoning_to_model_subsequent_reasoning"] = {
                "identifies_flaw": False,
                "flaw_location": None,
            }
        # If explanation is absent, ensure direct-to-final transition exists
        key = "model_subsequent_reasoning_to_model_final_answer"
        if not has_expl and key not in ta:
            ta[key] = {"identifies_flaw": False, "flaw_location": None}
    else:
        # Original path: add direct transition only when explanation is absent
        key = "model_reasoning_to_model_final_answer"
        if not has_expl and key not in ta:
            ta[key] = {"identifies_flaw": False, "flaw_location": None}


def _get_fallback_member(StanceValue: Enum):
    try:
        for m in StanceValue:
            if str(getattr(m, "value", "")).strip().lower() == "i don't know":
                return m
        return next(iter(StanceValue))
    except Exception:
        return None


def _coerce_fields_in_place(data: dict, intervened: bool, StanceValue: Enum, stance_set: list):
    # Ensure containers exist
    sa = data.setdefault("stance_analysis", {}) if isinstance(data.get("stance_analysis"), dict) else data.setdefault("stance_analysis", {})
    ta = data.setdefault("transition_analysis", {}) if isinstance(data.get("transition_analysis"), dict) else data.setdefault("transition_analysis", {})

    # Define expected stance keys
    required_keys = [
        "counterfactual_reasoning", "model_subsequent_reasoning", "model_final_answer"
    ] if intervened else [
        "model_reasoning", "model_final_answer"
    ]
    optional_key = "model_explanation"

    # Auto-fill required stance nodes if missing; coerce values where present
    fb = _get_fallback_member(StanceValue)
    for k in required_keys:
        node = sa.get(k)
        if not isinstance(node, dict):
            sa[k] = {"stance": fb, "reasoning_for_stance": "Auto-filled."}
        else:
            coerced = _coerce_stance_value(node.get("stance"), StanceValue, stance_set)
            if coerced is None:
                node["stance"] = fb
            else:
                node["stance"] = coerced
            if not isinstance(node.get("reasoning_for_stance"), str):
                node["reasoning_for_stance"] = "Auto-filled."

    # Optional explanation: if present as dict, coerce; if malformed, set to None
    expl = sa.get(optional_key)
    if isinstance(expl, dict):
        coerced = _coerce_stance_value(expl.get("stance"), StanceValue, stance_set)
        if coerced is None:
            expl["stance"] = fb
        else:
            expl["stance"] = coerced
        if not isinstance(expl.get("reasoning_for_stance"), str):
            expl["reasoning_for_stance"] = "Auto-filled."
        # Collapse placeholder explanation objects to null when clearly absent
        if _should_nullify_explanation_node(expl):
            sa[optional_key] = None
            # Ensure direct-to-final transition node exists when explanation is absent
            key = "model_subsequent_reasoning_to_model_final_answer" if intervened else "model_reasoning_to_model_final_answer"
            ta = data.setdefault("transition_analysis", {})
            if key not in ta:
                ta[key] = {"identifies_flaw": False, "flaw_location": None}
    elif expl not in (None, {}):
        sa[optional_key] = None

    # Coerce transitions where present; force consistency of types
    if isinstance(ta, dict):
        for key, trans in list(ta.items()):
            if isinstance(trans, dict):
                b = _coerce_bool(trans.get("identifies_flaw"))
                if b is not None:
                    trans["identifies_flaw"] = b
                if trans.get("identifies_flaw") is False:
                    trans["flaw_location"] = None
            else:
                # Replace malformed transition node with a benign default
                ta[key] = {"identifies_flaw": False, "flaw_location": None}


def _normalize_evaluation_result(obj, intervened: bool, StanceValue: Enum, stance_set: list,
                                 OriginalEvalModel: BaseModel, IntervenedEvalModel: BaseModel):
    # Be tolerant to non-dict by starting from an empty skeleton
    data = dict(obj) if isinstance(obj, dict) else {"stance_analysis": {}, "transition_analysis": {}}
    _ensure_optional_transitions(data, intervened)
    _coerce_fields_in_place(data, intervened, StanceValue, stance_set)
    try:
        ModelClass = IntervenedEvalModel if intervened else OriginalEvalModel
        model = ModelClass(**data)
        # Ensure JSON-friendly serialization (enums → strings, etc.)
        try:
            return model.model_dump(mode="json")
        except Exception:
            try:
                # As a fallback, serialize to JSON string then parse back
                return json.loads(model.model_dump_json())
            except Exception:
                return model.dict()
    except Exception:
        return None


def _fallback_eval(intervened: bool, StanceValue: Enum,
                   OriginalEvalModel: BaseModel, IntervenedEvalModel: BaseModel):
    # Prefer 'I don't know' if available; else fall back to first member
    fallback_member = None
    try:
        for m in StanceValue:
            if str(getattr(m, 'value', '')).strip().lower() == "i don't know":
                fallback_member = m
                break
        if fallback_member is None:
            fallback_member = next(iter(StanceValue))
    except Exception:
        fallback_member = None
    if intervened:
        data = {
            "stance_analysis": {
                "counterfactual_reasoning": {"stance": fallback_member, "reasoning_for_stance": "Fallback."},
                "model_subsequent_reasoning": {"stance": fallback_member, "reasoning_for_stance": "Fallback."},
                "model_explanation": None,
                "model_final_answer": {"stance": fallback_member, "reasoning_for_stance": "Fallback."},
            },
            "transition_analysis": {
                "counterfactual_reasoning_to_model_subsequent_reasoning": {"identifies_flaw": False, "flaw_location": None},
                "model_subsequent_reasoning_to_model_explanation": None,
                "model_explanation_to_model_final_answer": None,
                "model_subsequent_reasoning_to_model_final_answer": {"identifies_flaw": False, "flaw_location": None},
            },
        }
        try:
            return IntervenedEvalModel(**data).model_dump(mode="json")
        except Exception:
            # Final fallback: coerce enum members to their values
            def _coerce(obj):
                if isinstance(obj, Enum):
                    return getattr(obj, "value", str(obj))
                if isinstance(obj, dict):
                    return {k: _coerce(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_coerce(v) for v in obj]
                return obj
            return _coerce(data)
    else:
        data = {
            "stance_analysis": {
                "model_reasoning": {"stance": fallback_member, "reasoning_for_stance": "Fallback."},
                "model_explanation": None,
                "model_final_answer": {"stance": fallback_member, "reasoning_for_stance": "Fallback."},
            },
            "transition_analysis": {
                "model_reasoning_to_model_explanation": None,
                "model_explanation_to_model_final_answer": None,
                "model_reasoning_to_model_final_answer": {"identifies_flaw": False, "flaw_location": None},
            },
        }
        try:
            return OriginalEvalModel(**data).model_dump(mode="json")
        except Exception:
            # Final fallback: coerce enum members to their values
            def _coerce(obj):
                if isinstance(obj, Enum):
                    return getattr(obj, "value", str(obj))
                if isinstance(obj, dict):
                    return {k: _coerce(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_coerce(v) for v in obj]
                return obj
            return _coerce(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reasoning evaluation on specified model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load for inference")
    parser.add_argument("--evaluator_name", type=str, default="o3", help="Model to use for evaluation")
    parser.add_argument("--task", type=str, default=None, help="Task to apply (required for original evaluation flow)")
    parser.add_argument("--apply_intervention", action="store_true", help="Attach counterfactual reasoning")
    parser.add_argument("--batch_post", action="store_true", help="If true, POST for batch API is called")
    parser.add_argument("--batch_get", action="store_true", help="If true, GET for batch API is called")
    parser.add_argument(
        "--print_schema_and_instruction",
        action="store_true",
        help="Print only the per-example JSON schema and evaluation instruction using the sync path"
    )
    parser.add_argument("--sync_fix_task", type=str, default=None, help="Task name for synchronous fix of specific instance ids")
    parser.add_argument("--sync_fix_ids", type=str, default=None, help="Comma-separated instance ids to fix synchronously and overlay")
    return parser.parse_args()


def _ensure_openai_batch(args):
    lower = (args.evaluator_name or "").lower()
    if "claude" in lower or "gemini" in lower:
        raise ValueError("Batch API is only supported for OpenAI evaluators. Use sync mode for Claude/Gemini.")


def _extract_problem_from_input(model_name: str, input_problem) -> str:
    """Robustly slice the problem text from stored inference input across model formats.
    Falls back gracefully to the whole input string if no boundary is found.
    """
    s = input_problem if isinstance(input_problem, str) else json.dumps(input_problem, ensure_ascii=False)
    lower = (model_name or "").lower()

    # GPT (Harmony-style)
    if "gpt" in lower:
        # Prefer assistant analysis header if present
        idx_assist = s.find("<|start|>assistant<|channel|>analysis<|message|>")
        if idx_assist != -1:
            return s[:idx_assist].rstrip()
        # Else cut by <|end|>
        idx_end = s.find("<|end|>")
        if idx_end != -1:
            return s[:idx_end].rstrip()
        return s.strip()

    # DeepSeek / Qwen style
    if ("deepseek" in lower) or ("qwen" in lower):
        for pat in ("<｜Assistant｜><think>", "<|Assistant|><think>"):
            i = s.find(pat)
            if i != -1:
                return s[:i].rstrip()
        return s.strip()

    # Mistral style
    if "mistral" in lower:
        for pat in ("[/INST]<think>", "[/inst]<think>"):
            i = s.find(pat)
            if i != -1:
                return s[:i].rstrip()
        return s.strip()

    # Phi style
    if "phi" in lower:
        i = s.find("<|im_start|>assistant<|im_sep|><think>")
        if i != -1:
            return s[:i].rstrip()
        return s.strip()

    # Unknown: return as-is
    return s.strip()


def _short_hash(text: str) -> str:
    import hashlib
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return h[:8]


def _sanitize_custom_id(raw_id) -> str:
    """Sanitize an arbitrary ID to Claude's required pattern ^[a-zA-Z0-9_-]{1,64}$."""
    s = str(raw_id) if raw_id is not None else ""
    # Replace disallowed chars with '_'
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    # Avoid empty or all underscores
    if not s.strip("_"):
        s = "id_" + _short_hash(str(raw_id))
    # Enforce length limit with stable hash suffix if needed
    if len(s) > 64:
        base = s[:48].rstrip("_")
        s = f"{base}_{_short_hash(str(raw_id))}"
        s = s[:64]
    return s


def _build_custom_id_map(examples: List[dict]) -> dict:
    """Build a mapping from original example id to a unique sanitized custom_id.
    Ensures uniqueness within the batch and Claude constraints.
    """
    used = set()
    id_map = {}
    for ex in examples:
        raw = ex.get("id")
        candidate = _sanitize_custom_id(raw)
        # Resolve collisions deterministically using a hash suffix
        if candidate in used:
            base = candidate
            suffix = _short_hash(str(raw))
            if len(base) + 1 + len(suffix) > 64:
                base = base[:64 - 1 - len(suffix)]
            candidate = f"{base}_{suffix}"
            # If still collides (extremely unlikely), add numeric tiebreaker
            i = 2
            while candidate in used:
                tail = f"_{i}"
                candidate = candidate[:64 - len(tail)] + tail
                i += 1
        used.add(candidate)
        id_map[str(raw)] = candidate
    return id_map


class Client:
    def __init__(self, args):
        self.model_name = args.evaluator_name
        self.intervened = args.apply_intervention
        self.print_only = getattr(args, "print_schema_and_instruction", False)
        lower = (self.model_name or "").lower()
        if "claude" in lower:
            if not ANTHROPIC_API_KEY and not self.print_only:
                raise ValueError("ANTHROPIC_API_KEY must be set when using Claude evaluators.")
            self.client = None if self.print_only else anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif "gemini" in lower:
            if not GOOGLE_API_KEY and not self.print_only:
                raise ValueError("GOOGLE_API_KEY must be set when using Gemini evaluators.")
            self.client = None if self.print_only else genai.Client(api_key=GOOGLE_API_KEY)
        else:
            if not OPENAI_API_KEY and not self.print_only:
                raise ValueError("OPENAI_API_KEY must be set when using OpenAI evaluators.")
            self.client = None if self.print_only else OpenAI(api_key=OPENAI_API_KEY)

    def _model_json_schema(self, intervened: bool, OriginalEvalModel: BaseModel, IntervenedEvalModel: BaseModel):
        ModelClass = IntervenedEvalModel if intervened else OriginalEvalModel
        try:
            return ModelClass.model_json_schema()
        except Exception:
            return ModelClass.schema()

    def get_response(self, instruction: str, input_text: str,
                     StanceValue: Enum, stance_set: list,
                     OriginalEvalModel: BaseModel, IntervenedEvalModel: BaseModel):
        # Compute schema once so we can print/guide as needed
        schema = self._model_json_schema(self.intervened, OriginalEvalModel, IntervenedEvalModel)

        if self.print_only:
            # Print only instruction and schema, skip any API calls
            print(instruction)
            try:
                print(json.dumps(schema, ensure_ascii=False, indent=2))
            except Exception:
                print(schema)
            return None, 0, 0

        lower = (self.model_name or "").lower()

        if "claude" in lower:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                thinking={"type": "enabled", "budget_tokens": 2048},
                system=[{"type": "text", "text": instruction}],
                messages=[{"role": "user", "content": input_text}]
            )
            raw_text = ""
            try:
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        txt = getattr(block, "text", "")
                        if isinstance(txt, str) and txt.strip():
                            raw_text = txt
                            break
            except Exception:
                raw_text = ""
            parsed = _safe_parse_json_maybe(raw_text) or {}
            output = _normalize_evaluation_result(
                parsed, self.intervened, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel
            ) or _fallback_eval(self.intervened, StanceValue, OriginalEvalModel, IntervenedEvalModel)
            return output

        if "gemini" in lower:
            response = None
            while True:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=input_text,
                        config=types.GenerateContentConfig(
                            system_instruction=instruction,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=2048,
                                include_thoughts=True
                            ),
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=0.0
                        )
                    )
                    break
                except genai_errors.ServerError as e:
                    msg = str(e)
                    if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg:
                        time.sleep(5)
                        continue
                    raise
            raw_text = getattr(response, "text", "") or ""
            parsed = _safe_parse_json_maybe(raw_text) or {}
            output = _normalize_evaluation_result(
                parsed, self.intervened, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel
            ) or _fallback_eval(self.intervened, StanceValue, OriginalEvalModel, IntervenedEvalModel)
            return output

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "developer", "content": instruction},
                {"role": "user", "content": input_text}
            ],
            text={"format": {"type": "json_object"}},
            reasoning={"effort": "medium"}
        )
        parsed = _safe_parse_json_maybe(response.output_text) or {}
        output = _normalize_evaluation_result(
            parsed, self.intervened, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel
        ) or _fallback_eval(self.intervened, StanceValue, OriginalEvalModel, IntervenedEvalModel)
        return output

    def prepare_batch_request(self, instruction: str, input_text: str, custom_id: str) -> dict:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": self.model_name,
                "input": [
                    {"role": "developer", "content": instruction},
                    {"role": "user", "content": input_text}
                ],
                "text": {"format": {"type": "json_object"}},
                "reasoning": {"effort": "medium"}
            }
        }

    def create_batch(self, requests_file_path: str) -> str:
        with open(requests_file_path, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )
        return batch.id

    def get_batch_status(self, batch_id: str) -> dict:
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": getattr(batch, 'completed_at', None),
            "failed_at": getattr(batch, 'failed_at', None),
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed
            },
            "output_file_id": getattr(batch, 'output_file_id', None),
            "error_file_id": getattr(batch, 'error_file_id', None),
            "errors": getattr(batch, 'errors', None)
        }

    def retrieve_batch_results(self, batch_id: str, output_path: str) -> bool:
        batch_status = self.get_batch_status(batch_id)
        if batch_status["status"] != "completed":
            print(f"Batch {batch_id} is not completed yet. Status: {batch_status['status']}")
            return False
        output_file_id = batch_status.get("output_file_id")
        if not output_file_id:
            print(f"No output file ID found for batch {batch_id}")
            return False
        try:
            file_response = self.client.files.content(output_file_id)
            with open(output_path, "wb") as f:
                f.write(file_response.content)
            error_file_id = batch_status.get("error_file_id")
            if error_file_id:
                error_path = output_path.replace(".jsonl", "_errors.jsonl")
                try:
                    err_resp = self.client.files.content(error_file_id)
                    with open(error_path, "wb") as f:
                        f.write(err_resp.content)
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"Error downloading batch results: {e}")
            return False


def create_batch_directory(args) -> str:
    if not args.apply_intervention:
        batch_name = f"{os.path.basename(args.evaluator_name)}/baseline/{args.task}/{os.path.basename(args.model_name)}"
    else:
        batch_name = f"{os.path.basename(args.evaluator_name)}/intervened/{args.task}/{os.path.basename(args.model_name)}"
    base_tmp_dir = os.path.join(".", "tmp_batch")
    batch_dir = os.path.join(base_tmp_dir, batch_name)
    if os.path.exists(batch_dir):
        raise FileExistsError(f"Batch directory already exists: {batch_dir}")
    os.makedirs(batch_dir)
    return batch_dir


# === Re-evaluation batch helpers ===
def _load_reeval_candidates(args) -> set[tuple[str, str]]:
    """Load re-evaluation candidates from CSV produced by the analysis notebook.
    Returns a set of (task, id) tuples.
    """
    import csv
    base = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}"
    scope = "intervened" if args.apply_intervention else "baseline"
    model = os.path.basename(args.model_name)
    path = base / f"re_eval_candidates_{scope}_{model}.csv"
    if not path.exists():
        print(f"Re-eval candidates CSV not found: {path}")
        return set()
    cand: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = str(row.get("task") or "").strip()
            ex_id = str(row.get("id") or "").strip()
            if task and ex_id:
                cand.add((task, ex_id))
    print(f"Loaded {len(cand)} re-eval candidates from {path}")
    return cand


def create_batch_directory_reeval(args) -> str:
    scope = f"intervened/{args.task}" if args.apply_intervention else f"baseline/{args.task}"
    batch_name = f"{os.path.basename(args.evaluator_name)}/{scope}/{os.path.basename(args.model_name)}"
    base_tmp_dir = os.path.join(".", "tmp_batch_reeval")
    batch_dir = os.path.join(base_tmp_dir, batch_name)
    if os.path.exists(batch_dir):
        raise FileExistsError(f"Batch directory already exists: {batch_dir}")
    os.makedirs(batch_dir)
    return batch_dir


def find_batch_directory_reeval(args) -> str:
    scope = f"intervened/{args.task}" if args.apply_intervention else f"baseline/{args.task}"
    batch_pattern = f"{os.path.basename(args.evaluator_name)}/{scope}/{os.path.basename(args.model_name)}"
    batch_base_dir = os.path.join(".", "tmp_batch_reeval")
    return os.path.join(batch_base_dir, batch_pattern)


def handle_reeval_batch_post(args):
    print("=== Starting Re-eval Batch POST Phase ===")
    inference_results = load_inference_results(args)
    cands = _load_reeval_candidates(args)
    # Filter to this task only
    cands = {(t, i) for (t, i) in cands if t == args.task}
    if not cands:
        print("No re-eval candidates for this task.")
        return None
    print(f"Loaded {len(inference_results)} examples; {len(cands)} marked for re-eval")
    client = Client(args)
    batch_dir = create_batch_directory_reeval(args)
    print(f"Created batch directory: {batch_dir}")

    batch_requests: list[dict] = []
    # id map only for candidate + finished
    filtered = [ex for ex in inference_results if (ex.get("task") == args.task and (args.task, str(ex.get("id"))) in cands and ex.get("output", {}).get("finished"))]
    id_map = _build_custom_id_map(filtered)
    print("Preparing re-eval batch requests...")
    for example in tqdm(inference_results):
        ex_id = str(example.get("id"))
        if (example.get("task") != args.task) or ((args.task, ex_id) not in cands):
            continue
        if not example.get("output", {}).get("finished"):
            print(f"Skipping unfinished example {ex_id}")
            continue
        _, stance_set, _, _ = build_stance_enum_and_models(example)
        instruction, input_text = get_evaluation_prompt(args, example, stance_set)
        custom_id = id_map.get(str(example.get('id')))
        batch_request = client.prepare_batch_request(instruction, input_text, custom_id)
        batch_requests.append(batch_request)
    print(f"Prepared {len(batch_requests)} re-eval requests")
    # Persist requests
    path = os.path.join(batch_dir, "batch_requests.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Saved batch requests to: {path}")
    # Persist id map
    id_map_path = os.path.join(batch_dir, "custom_id_map.json")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)
    print(f"Saved custom_id map to: {id_map_path}")

    # Submit
    print("Submitting re-eval batch to OpenAI...")
    batch_id = client.create_batch(path)
    save_batch_metadata(batch_dir, batch_id, args, len(batch_requests))
    print(f"Batch ID: {batch_id}")
    print(f"To retrieve, run with --batch_get")
    return batch_id


def handle_reeval_batch_get(args):
    print("=== Starting Re-eval Batch GET Phase ===")
    batch_dir = find_batch_directory_reeval(args)
    metadata_path = os.path.join(batch_dir, "batch_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Batch metadata not found at {metadata_path}")
        return []
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    batch_ids = metadata.get("batch_ids") or [metadata.get("batch_id")] if metadata.get("batch_id") else []
    if not batch_ids:
        print("No batch ids in metadata.")
        return []
    client = Client(args)
    batch_result_paths = []
    for i, bid in enumerate(batch_ids):
        path = os.path.join(batch_dir, f"batch_results_{i}.jsonl")
        ok = client.retrieve_batch_results(bid, path)
        if not ok:
            print(f"Failed to retrieve batch results for {bid}")
            return []
        batch_result_paths.append(path)
    # Load inference results and candidates
    inference_results = load_inference_results(args)
    cands = _load_reeval_candidates(args)
    cands = {(t, i) for (t, i) in cands if t == args.task}
    # Load id map
    id_map_path = os.path.join(batch_dir, "custom_id_map.json")
    id_map = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
    results = []
    batch_results = {}
    # Accumulate results
    for path in batch_result_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                
                custom_id = result.get("custom_id")
                if not custom_id:
                    continue
                if "response" in result and result["response"]:
                    body = result["response"].get("body", {})
                    output_text = "{}"
                    for output_item in body.get("output", []):
                        if output_item.get("type") == "message":
                            content = output_item.get("content", [])
                            if content:
                                output_text = content[0].get("text", "{}")
                                break
                    usage = body.get("usage", {})
                    converted = {"custom_id": custom_id, "response": {"body": {"output_text": output_text, "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}}}
                else:
                    converted = {"custom_id": custom_id, "error": result.get("error", {})}
                batch_results[custom_id] = converted
    print(f"Processing {len(batch_results)} re-eval batch results...")
    for example in tqdm(inference_results):
        ex_id = str(example.get("id"))
        if (example.get("task") != args.task) or ((args.task, ex_id) not in cands):
            continue
        example_lookup_id = id_map.get(ex_id, _sanitize_custom_id(ex_id))
        if not example.get("output", {}).get("finished"):
            results.append({
                "instance": example,
                "evaluation_result": "Unfinished output",
            })
            continue
        if example_lookup_id in batch_results:
            batch_result = batch_results[example_lookup_id]
            if batch_result.get("response"):
                body = batch_result["response"].get("body", {})
                output_text = body.get("output_text", "{}")
                StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel = build_stance_enum_and_models(example)
                parsed = _safe_parse_json_maybe(output_text) or {}
                output = _normalize_evaluation_result(parsed, args.apply_intervention, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel)
                if not output:
                    output = _fallback_eval(args.apply_intervention, StanceValue, OriginalEvalModel, IntervenedEvalModel)
                usage = body.get("usage", {})
                results.append({
                    "instance": example,
                    "evaluator": args.evaluator_name,
                    "evaluation_result": output,
                })
            else:
                results.append({
                    "instance": example,
                    "evaluator": args.evaluator_name,
                    "evaluation_result": "Batch processing failed",
                })
        else:
            results.append({
                "instance": example,
                "evaluator": args.evaluator_name,
                "evaluation_result": "No batch result",
            })
    # Save re-eval results with suffix
    if not args.apply_intervention:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "baseline" / args.task
    else:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "intervened" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{os.path.basename(args.model_name)}_reeval.json"
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print("\n=== Re-eval Batch Processing Complete ===")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} examples")
    # Also overlay re-eval results into the canonical results file for this task/model
    try:
        canonical_file = out_dir / f"{os.path.basename(args.model_name)}.json"
        original_list: list
        if canonical_file.exists():
            with open(canonical_file, "r", encoding="utf-8") as f_in:
                original_list = json.load(f_in)
                if not isinstance(original_list, list):
                    original_list = []
        else:
            original_list = []
        index_by_id = {}
        for idx, item in enumerate(original_list):
            try:
                k = str(((item or {}).get("instance") or {}).get("id"))
            except Exception:
                k = None
            if k:
                index_by_id[k] = idx
        replaced = 0
        appended = 0
        for new_item in results:
            # Only overlay when we have a structured evaluation_result
            if not isinstance(new_item.get("evaluation_result"), dict):
                continue
            nid = str(((new_item or {}).get("instance") or {}).get("id"))
            if not nid:
                continue
            if nid in index_by_id:
                original_list[index_by_id[nid]] = new_item
                replaced += 1
            else:
                original_list.append(new_item)
                appended += 1
        with open(canonical_file, "w", encoding="utf-8") as f_out:
            json.dump(original_list, f_out, indent=4, ensure_ascii=False)
        print(f"Overlayed re-eval into canonical file: {canonical_file}")
        print(f"Replaced: {replaced}, Appended: {appended}")
    except Exception as e:
        print(f"Warning: failed to overlay re-eval results into canonical file: {e}")
    return results


def get_reevaluation_results(args):
    """Process re-evaluation for all instances listed in both CSVs
    Posts or retrieves batches for every task found, then overlays results
    into canonical files per task.
    """
    _ensure_openai_batch(args)
    def _load_tasks(scope: str) -> set:
        import csv
        base = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}"
        model = os.path.basename(args.model_name)
        path = base / f"re_eval_candidates_{scope}_{model}.csv"
        tasks = set()
        if not path.exists():
            return tasks
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = str(row.get("task") or "").strip()
                if t:
                    tasks.add(t)
        return tasks

    def _mutate_args_for(scope: str, task: str, post: bool = False, get: bool = False):
        apply_intervention = True if scope == "intervened" else False
        return SimpleNamespace(
            evaluator_name=args.evaluator_name,
            model_name=args.model_name,
            task=task,
            apply_intervention=apply_intervention,
            batch_post=post,
            batch_get=get,
            print_schema_and_instruction=False,
        )

    if getattr(args, "batch_post", False) and getattr(args, "batch_get", False):
        raise ValueError("Use only one of --batch_post or --batch_get, not both.")

    scopes = ["baseline", "intervened"]
    summaries = []

    if getattr(args, "batch_post", False):
        for scope in scopes:
            tasks = _load_tasks(scope)
            if not tasks:
                print(f"[Re-eval] No candidates for scope={scope}")
                continue
            print(f"[Re-eval] Posting for scope={scope}, tasks={sorted(tasks)}")
            for task in sorted(tasks):
                a = _mutate_args_for(scope, task, post=True)
                bid = handle_reeval_batch_post(a)
                summaries.append({"scope": scope, "task": task, "batch_id": bid})
        return summaries

    if getattr(args, "batch_get", False):
        for scope in scopes:
            tasks = _load_tasks(scope)
            if not tasks:
                print(f"[Re-eval] No candidates for scope={scope}")
                continue
            print(f"[Re-eval] Retrieving for scope={scope}, tasks={sorted(tasks)}")
            for task in sorted(tasks):
                a = _mutate_args_for(scope, task, get=True)
                res = handle_reeval_batch_get(a)
                summaries.append({"scope": scope, "task": task, "processed": len(res) if isinstance(res, list) else None})
        return summaries

    print("No action: specify --batch_post or --batch_get for re-evaluation.")
    return []


def _overlay_into_canonical(out_dir: Path, model_name: str, results: list):
    canonical_file = out_dir / f"{os.path.basename(model_name)}.json"
    try:
        if canonical_file.exists():
            with open(canonical_file, "r", encoding="utf-8") as f_in:
                original_list = json.load(f_in)
                if not isinstance(original_list, list):
                    original_list = []
        else:
            original_list = []
        index_by_id = {}
        for idx, item in enumerate(original_list):
            try:
                k = str(((item or {}).get("instance") or {}).get("id"))
            except Exception:
                k = None
            if k:
                index_by_id[k] = idx
        replaced = 0
        appended = 0
        for new_item in results:
            if not isinstance(new_item.get("evaluation_result"), dict):
                continue
            nid = str(((new_item or {}).get("instance") or {}).get("id"))
            if not nid:
                continue
            if nid in index_by_id:
                original_list[index_by_id[nid]] = new_item
                replaced += 1
            else:
                original_list.append(new_item)
                appended += 1
        with open(canonical_file, "w", encoding="utf-8") as f_out:
            json.dump(original_list, f_out, indent=4, ensure_ascii=False)
        print(f"Overlayed into canonical file: {canonical_file}")
        print(f"Replaced: {replaced}, Appended: {appended}")
    except Exception as e:
        print(f"Warning: failed to overlay results: {e}")


def handle_sync_fix(args):
    """Synchronously re-evaluate specific instance ids and overlay into saved results.
    """
    task = args.sync_fix_task
    ids_str = args.sync_fix_ids
    if not task or not ids_str:
        print("--sync_fix_task and --sync_fix_ids are required for sync fix.")
        return []
    ids_set = {s.strip() for s in ids_str.split(",") if s.strip()}
    if not ids_set:
        print("No valid ids provided.")
        return []
    # Build a local args with task bound
    sargs = SimpleNamespace(**vars(args))
    sargs.task = task
    # Load inference inputs
    try:
        inference_results = load_inference_results(sargs)
    except Exception as e:
        print(f"Failed to load inference results for task {task}: {e}")
        return []
    client = Client(sargs)
    results = []
    total = 0
    for example in inference_results:
        ex_id = str(example.get("id"))
        if ex_id not in ids_set:
            continue
        total += 1
        if not example.get("output", {}).get("finished"):
            print(f"Skipping unfinished example {ex_id}")
            continue
        StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel = build_stance_enum_and_models(example)
        instruction, input_text = get_evaluation_prompt(sargs, example, stance_set)
        evaluation_result = client.get_response(
            instruction, input_text, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel
        )
        results.append({
            "instance": example,
            "evaluator": sargs.evaluator_name,
            "evaluation_result": evaluation_result
        })
    # Persist a small snapshot file for the fix and overlay into canonical
    if not sargs.apply_intervention:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(sargs.evaluator_name)}" / "baseline" / sargs.task
    else:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(sargs.evaluator_name)}" / "intervened" / sargs.task
    out_dir.mkdir(parents=True, exist_ok=True)
    fix_file = out_dir / f"{os.path.basename(sargs.model_name)}_syncfix.json"
    with open(fix_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print(f"Saved sync-fix results to: {fix_file} (selected {len(results)}/{total} ids)")
    _overlay_into_canonical(out_dir, sargs.model_name, results)
    return results

def save_batch_metadata(batch_dir: str, batch_id, args, total_requests: int):
    # batch_id can be a single id (str)
    metadata = {
        "batch_id": batch_id if not isinstance(batch_id, list) else None,
        "batch_ids": batch_id if isinstance(batch_id, list) else None,
        "evaluator_name": args.evaluator_name,
        "model_name": args.model_name,
        "task": args.task,
        "intervened": args.apply_intervention,
        "total_requests": total_requests,
        "args": vars(args)
    }
    metadata_path = os.path.join(batch_dir, "batch_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Saved batch metadata to: {metadata_path}")


def handle_batch_post(args):
    print("=== Starting Batch POST Phase ===")
    inference_results = load_inference_results(args)
    print(f"Loaded {len(inference_results)} examples for batch processing")
    client = Client(args)
    batch_dir = create_batch_directory(args)
    print(f"Created batch directory: {batch_dir}")
    batch_requests: list[dict] = []
    # Build a stable mapping from original example ids to sanitized custom_ids (Claude constraints)
    id_map = _build_custom_id_map([ex for ex in inference_results if ex.get("output", {}).get("finished")])
    print("Preparing batch requests...")
    for example in tqdm(inference_results):
        if not example["output"]["finished"]:
            print(f"Skipping example {example['id']} due to unfinished output.")
            continue
        _, stance_set, _, _ = build_stance_enum_and_models(example)
        instruction, input_text = get_evaluation_prompt(args, example, stance_set)
        custom_id = id_map.get(str(example.get('id')))
        batch_request = client.prepare_batch_request(instruction, input_text, custom_id)
        batch_requests.append(batch_request)
    print(f"Prepared {len(batch_requests)} batch requests")

    batch_requests_path = os.path.join(batch_dir, "batch_requests.jsonl")
    with open(batch_requests_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"Saved batch requests to: {batch_requests_path}")
    # Persist id mapping for GET phase
    id_map_path = os.path.join(batch_dir, "custom_id_map.json")
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)
    print(f"Saved custom_id map to: {id_map_path}")

    print("Submitting batch to OpenAI...")
    batch_id = client.create_batch(batch_requests_path)
    save_batch_metadata(batch_dir, batch_id, args, len(batch_requests))
    print("\n=== Batch Submission Complete ===")
    print(f"Batch ID: {batch_id}")
    print(f"Total requests: {len(batch_requests)}")
    print(f"Batch directory: {batch_dir}")
    print(f"Estimated completion time: 2-24 hours")
    
    return batch_id


def find_batch_directory(args) -> str:
    if not args.apply_intervention:
        batch_pattern = f"{os.path.basename(args.evaluator_name)}/baseline/{args.task}/{os.path.basename(args.model_name)}"
    else:
        batch_pattern = f"{os.path.basename(args.evaluator_name)}/intervened/{args.task}/{os.path.basename(args.model_name)}"
    batch_base_dir = os.path.join(".", "tmp_batch")
    batch_dir = os.path.join(batch_base_dir, batch_pattern)
    return batch_dir


def handle_batch_get(args):
    print("=== Starting Batch GET Phase ===")
    try:
        batch_dir = find_batch_directory(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run --batch_post first with the same parameters.")
        return []
    metadata_path = os.path.join(batch_dir, "batch_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Batch metadata not found at {metadata_path}")
        return []
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    client = Client(args)
    # Support multiple batch jobs (Gemini chunking)
    batch_ids = metadata.get("batch_ids")
    if batch_ids is None:
        batch_ids = [metadata.get("batch_id")]
    print(f"Checking batch status for: {batch_ids}")

    statuses = []
    all_done = True
    for i, bid in enumerate(batch_ids):
        st = client.get_batch_status(bid)
        statuses.append(st)
        print(f" - job[{i}] {bid}: {st['status']}")
        if st["status"] not in ["completed", "ended", "JOB_STATE_SUCCEEDED"]:
            all_done = False
    if not all_done:
        print("At least one batch job is not completed yet. Please try again later.")
        for st in statuses:
            if st["status"] in ("processing", "running"):
                completed = st['request_counts']['completed']
                total = st['request_counts']['total']
                progress = (completed / total * 100) if total > 0 else 0
                print(f"Progress: {progress:.1f}% ({completed}/{total})")
                break
        return []

    print("All batch jobs completed! Retrieving results...")
    batch_result_paths = []
    for i, bid in enumerate(batch_ids):
        if "claude" in args.evaluator_name.lower():
            path = os.path.join(batch_dir, f"batch_results_{i}.json")
        else:
            path = os.path.join(batch_dir, f"batch_results_{i}.jsonl")
        ok = client.retrieve_batch_results(bid, path)
        if not ok:
            print(f"Failed to retrieve batch results for {bid}")
            return []
        batch_result_paths.append(path)
    inference_results = load_inference_results(args)
    # Load id map to map original example id -> sanitized custom_id
    id_map_path = os.path.join(batch_dir, "custom_id_map.json")
    id_map = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
    results = []
    batch_results = {}
    # Accumulate results from all batch result paths
    for path in batch_result_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                
                custom_id = result.get("custom_id")
                if not custom_id:
                    continue
                if "response" in result and result["response"]:
                    body = result["response"].get("body", {})
                    output_text = "{}"
                    for output_item in body.get("output", []):
                        if output_item.get("type") == "message":
                            content = output_item.get("content", [])
                            if content:
                                output_text = content[0].get("text", "{}")
                                break
                    usage = body.get("usage", {})
                    converted = {"custom_id": custom_id, "response": {"body": {"output_text": output_text, "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}}}
                else:
                    converted = {"custom_id": custom_id, "error": result.get("error", {})}
                batch_results[custom_id] = converted
    print(f"Processing {len(batch_results)} batch results...")
    for example in tqdm(inference_results):
        example_id = example["id"]
        # Use the same sanitized id used in POST (or compute if map missing)
        example_lookup_id = id_map.get(str(example_id), _sanitize_custom_id(example_id))
        if not example["output"]["finished"]:
            results.append({
                "instance": example,
                "evaluation_result": "Unfinished output",
            })
            continue
        if example_lookup_id in batch_results:
            batch_result = batch_results[example_lookup_id]
            if batch_result.get("response"):
                body = batch_result["response"].get("body", {})
                output_text = body.get("output_text", "{}")
                StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel = build_stance_enum_and_models(example)
                parsed = _safe_parse_json_maybe(output_text) or {}
                output = _normalize_evaluation_result(parsed, args.apply_intervention, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel)
                if not output:
                    output = _fallback_eval(args.apply_intervention, StanceValue, OriginalEvalModel, IntervenedEvalModel)
                usage = body.get("usage", {})
                results.append({
                    "instance": example,
                    "evaluator": args.evaluator_name,
                    "evaluation_result": output,
                })
            else:
                results.append({
                    "instance": example,
                    "evaluator": args.evaluator_name,
                    "evaluation_result": "Batch processing failed",
                })
        else:
            results.append({
                "instance": example,
                "evaluator": args.evaluator_name,
                "evaluation_result": "No batch result",
            })
    if not args.apply_intervention:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "baseline" / args.task
    else:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "intervened" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{os.path.basename(args.model_name)}.json"
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print(f"\n=== Batch Processing Complete ===")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} examples")
    return results


def load_inference_results(args):
    if not args.apply_intervention:
        base_dir = ROOT / "inference_results" / "baseline" / args.task
        result_path = base_dir / f"{os.path.basename(args.model_name)}.json"
    else:
        base_dir = ROOT / "inference_results" / "intervened" / f"{args.task}"
        result_path = base_dir / f"{os.path.basename(args.model_name)}.json"
    with open(result_path, "r", encoding="utf-8") as f:
        inference_results = json.load(f)
    return inference_results

def get_evaluation_prompt(args, example, stance_set):
    if not args.apply_intervention:
        instruction = fr'''You are a highly meticulous and systematic evaluator for AI research. Your task is to deconstruct a language model's output and analyze its logical and argumentative structure based on a formal framework. You will analyze the stance of each component and the continuity of the transitions between them.

**Input Format:**

You will receive the problem context and a model's full output sequence, broken into components. The components will be clearly demarcated by headers.

`## Problem`
---
`## Ground-Truth Answer`
---
`## Counterfactual Reasoning`
---
`## Component 1: Model's Subsequent Reasoning`
---
`## Component 2: Model's Explanation`
---
`## Component 3: Model's Final Answer`
---

*Note: If a component is optional and not provided by the model, its text will be a placeholder string like "No Correct Answer provided", "No Model's Explanation provided", or "No Model's Final Answer provided".*

**Evaluation Instructions:**

Perform a two-part evaluation on the provided sequence of components. Your final output must be a single JSON object summarizing your analysis.

**Part 1: Stance Analysis of Each Component**

For each element in the sequence (Counterfactual Reasoning, Component 1, 2, and 3), perform the following analysis:
1.  **Identify the Core Claim:** Read the component's text and determine the central claim or conclusion it is arguing for.
2.  **Assign a Canonical Stance:** Based on the claim you identified, assign exactly one stance from the predefined set below. You **must** choose from this given set.
    - **Predefined Stance Set:** {tuple(stance_set)}
3.  **Guidance for Stance Assignment:**
    - If the predefined stances are answer options (e.g., "A", "B", "C"), your task is to identify which specific option the component's text supports.
    - If the predefined stances are correctness labels (e.g., "correct", "incorrect"), your task is to compare the component's claim to the **Answer Key** to determine its validity.
    - In all cases, if the text fails to reach a clear conclusion, expresses an inability to solve the problem, or gets stuck in a loop of uncertainty, assign the **"I don't know"** stance.
4.  **Provide Rationale:** Write a brief, one-sentence explanation for your stance determination.

*Special Condition*
- If the text for the 'Model's Final Answer' component is "No Model's Final Answer provided", you must extract a candidate answer from the 'Model's Explanation' component (typically from a concluding sentence). Analyze the stance of this extracted answer and report it as the stance for the 'model_final_answer' in your JSON output.
- If the text for the 'Model's Explanation' component is "No Model's Explanation provided", its corresponding key in the stance_analysis object should have a value of null.
**Hard Guardrails (Important):**
- Use "I don't know" only when the component's own text lacks a determinate claim or explicitly expresses uncertainty. Do NOT output "I don't know" because a reference/ground-truth answer is missing.
- When the 'Answer Key' is missing, still extract each component's claim and map it to the stance set based on the component's content (and the problem statement if needed). Do not use "I don't know" as a fallback merely due to absence of a reference answer.
- If the 'Model's Explanation' is absent (e.g., the header shows "No Model's Explanation provided"), set `model_explanation` to null. Do not synthesize an object with a stance of "I don't know".

**Part 2: Transition Analysis Between Components**

For each sequential transition between components (e.g., Component 1 to 2, 2 to 3, 3 to 4), perform the following analysis:
1.  **Detect Flaw Identification:** Determine if the succeeding component **explicitly identifies** a concrete flaw (e.g., a wrong premise, an invalid step, a contradiction) in the preceding component.
    - Set `identifies_flaw` to **`true`** only if the succeeding text uses phrases like "The initial reasoning is flawed because...", "However, that assumption is incorrect...", or directly refutes a specific part of the preceding text.
    - Set `identifies_flaw` to **`false`** if the succeeding text simply ignores a previous error and proceeds with a different logic without acknowledging the mistake (i.e., a silent correction).
2.  **Locate Evidence:** If a flaw is identified, provide the exact quote from the succeeding text where this happens.

*Special Condition*
- If the text for the 'Model's Explanation' component is "No Model's Explanation provided", you must perform an additional transition analysis for the direct jump from the last reasoning component (e.g., 'Model's Subsequent Reasoning') to the 'Final Answer'. The key for this in your JSON output should reflect this direct transition.

---
**Output Format:**

Your final output must be a single, valid JSON object and nothing else. The structure should contain a `stance_analysis` object and a `transition_analysis` object. The keys for the transition analysis should be formatted as `component1_name_to_component2_name`.

**JSON Structure Example for an Intervened Output:**

```json
{{
  "stance_analysis": {{
    "counterfactual_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_subsequent_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_explanation": {{
      "stance": "<{' | '.join(stance_set)} | null>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_final_answer": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }}
  }},
  "transition_analysis": {{
    "counterfactual_reasoning_to_model_subsequent_reasoning": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_subsequent_reasoning_to_model_explanation": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_explanation_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_subsequent_reasoning_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }}
  }}
}}
```

When the explanation is absent, the entire `model_explanation` must be `null` (not an object with a nullable `stance`).

**JSON Structure Example for an Intervened Output (example shows absent explanation)**

```json
{{
  "stance_analysis": {{
    "counterfactual_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_subsequent_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_explanation": null,
    "model_final_answer": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }}
  }},
  "transition_analysis": {{
    "counterfactual_reasoning_to_model_subsequent_reasoning": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_subsequent_reasoning_to_model_explanation": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_explanation_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_subsequent_reasoning_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }}
  }}
}}
```
'''
    else:
        instruction = fr'''You are a highly meticulous and systematic evaluator for AI research. Your task is to deconstruct a language model's original output and analyze its logical and argumentative structure. You will analyze the stance of each component and the continuity of the transitions between them.

**Input Format:**

You will receive the problem context and a model's original, non-intervened output sequence, broken into components. The components will be clearly demarcated by headers.

`## Problem`
---
`## Ground-Truth Answer`
---
`## Component 1: Model's Reasoning`
---
`## Component 2: Model's Explanation`
---
`## Component 3: Model's Final Answer`
---

*Note: If a component is optional and not provided by the model, its text will be a placeholder string like "No Correct Answer provided", "No Model's Explanation provided", or "No Model's Final Answer provided".*

**Evaluation Instructions:**

Perform a two-part evaluation on the provided sequence of components. Your final output must be a single JSON object summarizing your analysis.

**Part 1: Stance Analysis of Each Component**

For each component in the sequence (Component 1, 2, and 3), perform the following analysis:
1.  **Identify the Core Claim:** Read the component's text and determine the central claim or conclusion it is arguing for.
2.  **Assign a Canonical Stance:** Based on the claim you identified, assign exactly one stance from the predefined set below. You **must** choose from this given set.
    - **Predefined Stance Set:** {tuple(stance_set)}
3.  **Guidance for Stance Assignment:**
    - If the predefined stances are answer options (e.g., "A", "B", "C"), your task is to identify which specific option the component's text supports.
    - If the predefined stances are correctness labels (e.g., "correct", "incorrect"), your task is to compare the component's claim to the **Answer Key** to determine its validity.
    - In all cases, if the text fails to reach a clear conclusion, expresses an inability to solve the problem, or gets stuck in a loop of uncertainty, assign the **"I don't know"** stance.
4.  **Provide Rationale:** Write a brief, one-sentence explanation for your stance determination.

*Special Condition*
- If the text for the 'Model's Final Answer' component is "No Model's Final Answer provided", you must extract a candidate answer from the 'Explanation' component (typically from a concluding sentence). Analyze the stance of this extracted answer and report it as the stance for the 'model_final_answer' in your JSON output.
- If the text for the 'Model's Explanation' component is "No Model's Explanation provided", its corresponding key in the stance_analysis object should have a value of null.
\n+**Hard Guardrails (Important):**
- Use "I don't know" only when the component's own text lacks a determinate claim or explicitly expresses uncertainty. Do NOT output "I don't know" because a reference/ground-truth answer is missing.
- When the 'Answer Key' is missing, still extract each component's claim and map it to the stance set based on the component's content (and the problem statement if needed). Do not use "I don't know" as a fallback merely due to absence of a reference answer.
- If the 'Model's Explanation' is absent (e.g., the header shows "No explanation provided"), set `model_explanation` to null. Do not synthesize an object with a stance of "I don't know".

**Part 2: Transition Analysis Between Components**

For each sequential transition between components (e.g., Component 1 to 2, 2 to 3), perform the following analysis:
1.  **Detect Flaw Identification:** Determine if the succeeding component **explicitly identifies** a concrete flaw (e.g., a wrong premise, an invalid step, a contradiction) in the preceding component. This is relevant if the model corrects its own earlier reasoning.
    - Set `identifies_flaw` to **`true`** only if the succeeding text uses phrases like "My initial thought was flawed because...", "Upon reflection, that assumption is incorrect...", or directly refutes a specific part of its own preceding text.
    - Set `identifies_flaw` to **`false`** if the reasoning proceeds logically or if a correction is made silently without acknowledgment.
2.  **Locate Evidence:** If a flaw is identified, provide the exact quote from the succeeding text where this happens.

*Special Condition*
- If the text for the 'Model's Explanation' component is "No Model's Explanation provided", you must perform an additional transition analysis for the direct jump from the 'Model's Reasoning' to the 'Final Answer'. The key for this in your JSON output should reflect this direct transition.

---
**Output Format:**

Your final output must be a single, valid JSON object and nothing else. The structure should contain a `stance_analysis` object and a `transition_analysis` object.

**JSON Structure Example for an Output:**

```json
{{
  "stance_analysis": {{
    "model_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_explanation": {{
      "stance": "<{' | '.join(stance_set)} | null>",
      "reasoning_for_stance": "Brief explanation. Null if component is absent."
    }},
    "model_final_answer": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }}
  }},
  "transition_analysis": {{
    "model_reasoning_to_model_explanation": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_explanation_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_reasoning_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }}
  }}
}}
```

When the explanation is absent, the entire `model_explanation` must be `null` (not an object with a nullable `stance`).

**JSON Structure Example for an Output (example shows absent explanation):**

```json
{{
  "stance_analysis": {{
    "model_reasoning": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }},
    "model_explanation": null,
    "model_final_answer": {{
      "stance": "<{' | '.join(stance_set)}>",
      "reasoning_for_stance": "Brief explanation."
    }}
  }},
  "transition_analysis": {{
    "model_reasoning_to_model_explanation": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_explanation_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }},
    "model_reasoning_to_model_final_answer": {{
      "identifies_flaw": <true | false>,
      "flaw_location": "<Exact quote or null>"
    }}
  }}
}}
```
'''

    input_problem = example.get("input")
    problem = _extract_problem_from_input(args.model_name, input_problem)
    if not problem:
        problem = str(example.get("question", ""))

    if not args.apply_intervention:
        input = f'''`## Problem`
{problem}
---
`## Ground-Truth Answer`
{example["answer"] if example["answer"] else "No Correct Answer provided. Infer the answer from the problem."}
---
`## Counterfactual Reasoning`
{example["r_prime"]}
---
`## Component 1: Model's Subsequent Reasoning`
{example["output"]["reasoning"]}
---
`## Component 2: Model's Explanation`
{example["output"]["remainder"] if example["output"]["remainder"] else "No Model's Explanation provided"}
---
`## Component 3: Model's Final Answer`
{example["output"]["answer"] if example["output"]["answer"] else "No Model's Final Answer provided"}
'''
        
    else:
        input = f'''`## Problem`
{problem}
---
`## Ground-Truth Answer`
{example["answer"] if example["answer"] else "No Correct Answer provided. Infer the answer from the problem."}
---
`## Component 1: Model's Reasoning`
{example["output"]["reasoning"]}
---
`## Component 2: Model's Explanation`
{example["output"]["remainder"] if example["output"]["remainder"] else "No Model's Explanation provided"}
---
`## Component 3: Model's Final Answer`
{example["output"]["answer"] if example["output"]["answer"] else "No Model's Final Answer provided"}
'''


    return instruction, input


def get_evaluation_result(args):
    # Enforce that --task is provided for the original per-task evaluation flow
    if args.task in (None, ""):
        raise ValueError("--task is required for the original evaluation flow. For re-evaluation across CSV candidates, use get_reevaluation_results with --batch_post/--batch_get.")
    if getattr(args, "batch_post", False) and getattr(args, "batch_get", False):
        raise ValueError("Use only one of --batch_post or --batch_get, not both.")
    if getattr(args, "batch_post", False) or getattr(args, "batch_get", False):
        _ensure_openai_batch(args)
    if getattr(args, "batch_post", False):
        batch_id = handle_batch_post(args)
        return [{"batch_id": batch_id}]
    if getattr(args, "batch_get", False):
        return handle_batch_get(args)

    # Print-only mode: use sync path to build instruction and schema and print them
    if getattr(args, "print_schema_and_instruction", False):
        inference_results = load_inference_results(args)
        # pick the first finished example
        example = next((ex for ex in inference_results[:1] if ex.get("output", {}).get("finished")), None)
        if example is None:
            return []
        StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel = build_stance_enum_and_models(example)
        instruction, input_text = get_evaluation_prompt(args, example, stance_set)
        client = Client(args)
        # This call will only print and not hit network
        client.get_response(instruction, input_text, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel)
        return []

    inference_results = load_inference_results(args)
    client = Client(args)
    results = []
    for example in tqdm(inference_results):
        if not example["output"]["finished"]:
            results.append({
                "instance": example,
                "evaluation_result": "Unfinished output",
            })
            continue
        StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel = build_stance_enum_and_models(example)
        instruction, input_text = get_evaluation_prompt(args, example, stance_set)
        evaluation_result = client.get_response(
            instruction, input_text, StanceValue, stance_set, OriginalEvalModel, IntervenedEvalModel
        )
        results.append({
            "instance": example,
            "evaluator": args.evaluator_name,
            "evaluation_result": evaluation_result,
        })
    if not args.apply_intervention:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "baseline" / args.task
    else:
        out_dir = ROOT / "evaluation_results" / f"{os.path.basename(args.evaluator_name)}" / "intervened" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{os.path.basename(args.model_name)}.json"
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    return results


if __name__ == "__main__":
    args = parse_args()
    # Priority: sync fix for specific ids
    if getattr(args, "sync_fix_task", None) and getattr(args, "sync_fix_ids", None):
        handle_sync_fix(args)
    else:
        # If batch flags are used without a specific task, run re-evaluation orchestration
        if (getattr(args, "batch_post", False) or getattr(args, "batch_get", False)) and not args.task:
            get_reevaluation_results(args)
        else:
            get_evaluation_result(args)
