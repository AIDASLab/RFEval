from vllm import LLM, SamplingParams

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Role,
)

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import anthropic

import json
import os
import re
import time
import inspect
import sys as _sys
from tqdm import tqdm
import argparse

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

from pathlib import Path
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run reasoning inference on specified model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load for inference")
    parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs for inference")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="Portion of single GPU utilization for inference")
    parser.add_argument("--task", type=str, required=True, help="Task to apply")
    parser.add_argument("--dataset_repo", type=str, default="snu-aidas/RFEval", help="HF dataset repo id")
    parser.add_argument("--dataset_config", type=str, default=None, help="HF dataset config/subset name")
    parser.add_argument("--dataset_split", type=str, default="train", help="HF dataset split (default: train)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of dataset to run inference on")
    parser.add_argument("--end_idx", type=int, default=999999, help="End index of dataset to run inference on")
    parser.add_argument("--apply_intervention", action="store_true", help="Attach counterfactual reasoning")
    parser.add_argument(
        "--use-proprietary-api",
        dest="use_proprietary_api",
        action="store_true",
        help="Use proprietary API backends (OpenAI/Anthropic/Google) instead of local vLLM"
    )
    return parser.parse_args()

def load_dataset(args):
    from datasets import load_dataset as hf_load_dataset
    if not args.dataset_config:
        raise ValueError("--dataset_config is required (use a task name like code_generation)")
    try:
        return hf_load_dataset(args.dataset_repo, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        # Fallback for configs with nested JSON fields that Arrow can't infer
        fname = f"data/{args.dataset_config}.json"
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=args.dataset_repo,
                filename=fname,
                repo_type="dataset",
            )
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            raise e

def import_dataset_utils(example):
    if example["content"]["source"] in ["livecodebench_v6", "livecodebench_v5", "livecodebench_v4", "livecodebench_v3", "livecodebench_v2", "livecodebench_v1"]:
        from dataset_utils.livecodebench.utils import parsing_inference_input
    elif example["content"]["source"] == "ds1000":
        from dataset_utils.ds1000.utils import parsing_inference_input
    
    elif example["content"]["source"] in ["mmlu/college_mathematics", "mmlu_college_mathematics"]:
        from dataset_utils.mmlu.college_mathematics.utils import parsing_inference_input
    elif example["content"]["source"] in ["mmlu/high_school_mathematics", "mmlu_high_school_mathematics"]:
        from dataset_utils.mmlu.high_school_mathematics.utils import parsing_inference_input
    elif example["content"]["source"] == "gsm8k":
        from dataset_utils.gsm8k.utils import parsing_inference_input
    
    elif example["content"]["source"] == "prontoqa":
        from dataset_utils.prontoqa.utils import parsing_inference_input
    elif example["content"]["source"] == "rulebert_union_rules":
        from dataset_utils.rulebert_union_rules.utils import parsing_inference_input

    elif example["content"]["source"] == "scitab":
        from dataset_utils.scitab.utils import parsing_inference_input
    
    elif example["content"]["source"] == "pubmedqa":
        from dataset_utils.pubmedqa.utils import parsing_inference_input
    
    elif example["content"]["source"] in ["mmlu/professional_law", "mmlu_professional_law"]:
        from dataset_utils.mmlu.professional_law.utils import parsing_inference_input
    
    elif example["content"]["source"] == "peerread":
        from dataset_utils.peerread.utils import parsing_inference_input
    else:
        raise ValueError("Example is from unsupported dataset.")

    return parsing_inference_input

def _call_parsing_inference_input(parsing_inference_input, example, args):
    try:
        sig = inspect.signature(parsing_inference_input)
        if len(sig.parameters) >= 2:
            augmented_flag = "yes" if args.apply_intervention else "no"
            return parsing_inference_input(augmented_flag, example)
    except Exception:
        pass
    return parsing_inference_input(example)

def parsing_inference_output(raw_output):
    """
    Parsing order:
    1) Extract reasoning inside <think>...</think> or [THINK]...[/THINK].
       Everything after the closing tag is the initial remainder.
    2) From that remainder, extract <answer>...</answer> or [answer]...[/answer].
       Remove the extracted span from the remainder.
    3) If step 2 fails, try heuristic candidates (labels, boxed, code).
       Remove the chosen span from the remainder.
    4) If no candidate is found, leave answer=None and keep all post-think text as remainder.
    """
    def _candidates_from_code_blocks(text):
        # ```python ...``` or ```py ...``` or ```cpp ...```
        cands = []
        for m in re.finditer(r"```(?:python|py|cpp)\s*([\s\S]*?)```", text, re.IGNORECASE):
            inner = m.group(1).strip()
            if inner:
                cands.append((inner, m.span()))
        return cands

    def _candidates_from_boxed(text):
        """
        Find \boxed{...} and return inner content with correct brace balancing.
        Handles nested braces like \boxed{\text{D: I and II only}} and \boxed{\frac{15}{64}}.
        Returns list of (inner_text, (start_index, end_index_of_full_span)).
        """
        cands = []
        for m in re.finditer(r"\\boxed\s*", text):
            p = m.end()
            # skip whitespace
            while p < len(text) and text[p].isspace():
                p += 1
            # need an unescaped '{'
            if p >= len(text) or text[p] != '{':
                continue
            # scan with brace depth
            depth = 1
            i = p + 1
            close = None
            while i < len(text):
                ch = text[i]
                prev = text[i - 1] if i > 0 else ''
                if ch == '{' and prev != '\\':
                    depth += 1
                elif ch == '}' and prev != '\\':
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
                i += 1
            if close is not None:
                inner = text[p + 1:close].strip()
                if inner:
                    cands.append((inner, (m.start(), close + 1)))
        return cands

    def _candidates_from_answer_labels(text):
        """
        Collect answers following '**Answer:**' / 'Answer:' or '**Final Answer:**' / 'Final Answer:'.
        End at a boundary: blank line, markdown heading, code fence, horizontal rule, list item,
        next known section label at line start, tag, or end of text.
        """
        cands = []
        label_core = r"(?:\*\*\s*)?(?:final\s*answer|answer|decision)(?:\s*\*\*)?\s*:\s*"
        for m in re.finditer(label_core, text, re.IGNORECASE):
            start = m.end()
            boundary_pat = re.compile(
                r"(?:"
                r"\n\s*\n"                                      # blank line
                r"|^\s*#{1,6}\s+.*$"                            # markdown heading
                r"|^\s*```"                                     # code fence start
                r"|^\s*(?:-{3,}|\*{3,}|_{3,})\s*$"              # horizontal rule
                r"|^\s*(?:[-*+]\s+|\d+[\.)]\s+)"                 # list item
                r"|^\s*(?:Explanation|Reasoning|Analysis|Solution|Notes?|Reference|References|Proof|Derivation|Calculation|Discussion|Conclusion|Summary|설명|해설|풀이|증명|참고|결론|요약)\s*:"  # optional labeled sections (EN/KR)
                r"|(?:</answer>|\[/answer\]|<think>|\[THINK\]|\[/THINK\]|</think>)" # tags
                r"|\Z"
                r")",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            b = boundary_pat.search(text, pos=start)
            end = b.start() if b else len(text)
            inner = text[start:end].strip()
            if inner:
                # include label in the removable span
                cands.append((inner, (m.start(), end)))
        return cands

    def _candidates_from_correct_answer_phrase(text):
        """
        Find phrases like: The correct answer is **D** (case-insensitive).
        Returns list of (inner_text, (start_idx, end_idx_of_full_span)).
        """
        cands = []
        pat = re.compile(r"the\s+correct\s+answer\s+is\s*:?\s*\*\*(.+?)\*\*", re.IGNORECASE | re.DOTALL)
        for m in pat.finditer(text):
            inner = (m.group(1) or "").strip()
            if inner:
                cands.append((inner, m.span()))
        return cands

    reasoning = None
    remainder = None
    answer = None

    # 1) Extract ALL <think>...</think> (or [THINK]...[/THINK]) blocks for reasoning,
    #    and remove them from the text to form the post-think remainder candidate.
    think_pat = re.compile(r"(?:<|\[)think(?:>|\])\s*(.*?)(?:<|\[)/think(?:>|\])", re.IGNORECASE | re.DOTALL)
    think_blocks = list(think_pat.finditer(raw_output))
    if think_blocks:
        parts = [m.group(1).strip() for m in think_blocks if m.group(1).strip()]
        reasoning = ("\n\n".join(parts)).strip() if parts else None
        post_think = think_pat.sub("", raw_output)
    else:
        post_think = raw_output  # no visible <think>; treat whole as post-think text

    # 2) Try explicit <answer>...</answer> within post-think text
    ans_tag_pat = re.compile(r"(?:<|\[)answer(?:>|\])\s*(.*?)(?:<|\[)/answer(?:>|\])", re.IGNORECASE | re.DOTALL)
    m_ans = ans_tag_pat.search(post_think)
    if m_ans:
        answer = m_ans.group(1).strip()
        # remove that span from the post-think remainder
        rem = (post_think[:m_ans.start()] + post_think[m_ans.end():]).strip()
        remainder = rem or None
    else:
        # 3) Heuristic candidates within post-think text
        candidates = []
        candidates += _candidates_from_answer_labels(post_think)
        candidates += _candidates_from_correct_answer_phrase(post_think)
        candidates += _candidates_from_boxed(post_think)
        candidates += _candidates_from_code_blocks(post_think)

        if candidates:
            cand_text, (s, e) = max(candidates, key=lambda x: x[1][1])  # prefer the last-ending candidate
            # If the chosen span starts with an Answer/Final Answer/Decision label,
            # and the inner text begins with a short choice token (A-E/Yes/No/True/False),
            # keep only that token as answer and push the rest back into remainder.
            sub = post_think[s:e]
            label_re = re.compile(r"^(?:\*\*\s*)?(?:final\s*answer|answer|decision)(?:\s*\*\*)?\s*:\s*", re.IGNORECASE)
            mlabel = label_re.match(sub)
            if mlabel:
                mchoice = re.match(r"^(?:\*\*\s*)?(?P<val>([A-E])|(yes|no|true|false))\b[\s\.:\-\)\]]*", cand_text, re.IGNORECASE)
                if mchoice:
                    short_ans = mchoice.group('val')
                    leftover = cand_text[mchoice.end():].strip()
                    answer = short_ans
                    rem = (post_think[:s] + (leftover if leftover else "") + post_think[e:]).strip()
                    remainder = rem or None
                else:
                    answer = cand_text
                    rem = (post_think[:s] + post_think[e:]).strip()
                    remainder = rem or None
            else:
                answer = cand_text
                rem = (post_think[:s] + post_think[e:]).strip()
                remainder = rem or None
        else:
            # 4) No answer found; keep all post-think as remainder
            remainder = post_think.strip() or None

    # Special-case refinement: if nothing remains outside and we only have
    # an <answer> block that still mixes explanation + final, try to re-parse
    # inside the answer text using the same heuristics (labels/boxed/code),
    # but WITHOUT re-reading <answer> tags. If that yields a candidate, adopt it.
    if (remainder is None or remainder == "") and isinstance(answer, str) and answer:
        inner_candidates = []
        inner_candidates += _candidates_from_answer_labels(answer)
        inner_candidates += _candidates_from_correct_answer_phrase(answer)
        inner_candidates += _candidates_from_boxed(answer)
        inner_candidates += _candidates_from_code_blocks(answer)

        if inner_candidates:
            inner_text, (s, e) = max(inner_candidates, key=lambda x: x[1][1])
            # Compute inner remainder relative to the answer text
            inner_rem = (answer[:s] + answer[e:]).strip()
            # Update to refined parse results
            answer = inner_text
            remainder = inner_rem or None

    return {
        "reasoning": reasoning,
        "remainder": remainder,
        "answer": answer,
    }

def parsing_answer_only(raw_output):
    """
    Given a string that is expected to predominantly contain the final answer,
    try to extract a clean answer and an optional remainder.

    Design choice:
    - If nothing is recognized (no tags, no common labels), we treat the WHOLE input
      as the remainder and set answer=None. This matches the caller's expectation that
      an unparsed segment should remain in remainder.
    """
    # 0) Explicit answer tags
    tag_pattern = r"(?:<|\[)answer(?:>|\])\s*(.*?)(?:<|\[)/answer(?:>|\])"
    m = re.search(tag_pattern, raw_output, re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip()
        start, end = m.span()
        remainder = (raw_output[:start] + raw_output[end:]).strip()
    else:
        ans = None
        remainder = None

        candidates = []  # list of (answer_text, (start, end))

        # 1) Code blocks: ```python ...``` or ```py ...``` or ```cpp ...```
        for m in re.finditer(r"```(?:python|py|cpp)\s*([\s\S]*?)```", raw_output, re.IGNORECASE):
            inner = m.group(1).strip()
            if inner:
                candidates.append((inner, m.span()))

        # 2) LaTeX \boxed{ ... } with balanced braces
        for m in re.finditer(r"\\boxed\s*", raw_output):
            p = m.end()
            while p < len(raw_output) and raw_output[p].isspace():
                p += 1
            if p >= len(raw_output) or raw_output[p] != '{':
                continue
            depth = 1
            i = p + 1
            close = None
            while i < len(raw_output):
                ch = raw_output[i]
                prev = raw_output[i - 1] if i > 0 else ''
                if ch == '{' and prev != '\\':
                    depth += 1
                elif ch == '}' and prev != '\\':
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
                i += 1
            if close is not None:
                inner = raw_output[p + 1:close].strip()
                if inner:
                    candidates.append((inner, (m.start(), close + 1)))

        # 3) Labels: Answer / Final Answer / Decision (robust boundary)
        label_core = r"(?:\*\*\s*)?(?:final\s*answer|answer|decision)(?:\s*\*\*)?\s*:\s*"
        occurrences = list(re.finditer(label_core, raw_output, re.IGNORECASE))
        if occurrences:
            # Use a generalized boundary to stop open-ended capture
            boundary_pat = re.compile(
                r"(?:"
                r"\n\s*\n"                                      # blank line
                r"|^\s*#{1,6}\s+.*$"                            # markdown heading
                r"|^\s*```"                                     # code fence start
                r"|^\s*(?:-{3,}|\*{3,}|_{3,})\s*$"              # horizontal rule
                r"|^\s*(?:[-*+]\s+|\d+[\.)]\s+)"                 # list item
                r"|^\s*(?:Explanation|Reasoning|Analysis|Solution|Notes?|Reference|References|Proof|Derivation|Calculation|Discussion|Conclusion|Summary|설명|해설|풀이|증명|참고|결론|요약)\s*:"  # optional labeled sections (EN/KR)
                r"|(?:</answer>|\[/answer\]|<think>|\[THINK\]|\[/THINK\]|</think>)" # tags
                r"|\Z"
                r")",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            # Prefer last occurrence; if that yields empty, also try first
            for idx in (len(occurrences) - 1, 0):
                m2 = occurrences[idx]
                start = m2.end()
                b = boundary_pat.search(raw_output, pos=start)
                end = b.start() if b else len(raw_output)
                inner = raw_output[start:end].strip()
                if inner:
                    # include the label in the removable span
                    candidates.append((inner, (m2.start(), end)))
                    break  # only keep one of last/first to avoid duplicates

        # 4) The correct answer is **X** (case-insensitive)
        for m in re.finditer(r"the\s+correct\s+answer\s+is\s*:?\s*\*\*(.+?)\*\*", raw_output, re.IGNORECASE | re.DOTALL):
            inner = (m.group(1) or "").strip()
            if inner:
                candidates.append((inner, m.span()))

        # 5) Choose candidate that ends latest
    if candidates:
        ans, (s, e) = max(candidates, key=lambda x: x[1][1])
        # If the chosen span starts with an Answer/Final Answer/Decision label,
        # and the inner text begins with a short choice token (A-E/Yes/No/True/False),
        # keep only that token as answer and push the rest back into remainder.
        sub = raw_output[s:e]
        label_re = re.compile(r"^(?:\*\*\s*)?(?:final\s*answer|answer|decision)(?:\s*\*\*)?\s*:\s*", re.IGNORECASE)
        mlabel = label_re.match(sub)
        if mlabel:
            mchoice = re.match(r"^(?:\*\*\s*)?(?P<val>([A-E])|(yes|no|true|false))\b[\s\.:\-\)\]]*", ans, re.IGNORECASE)
            if mchoice:
                short_ans = mchoice.group('val')
                leftover = ans[mchoice.end():].strip()
                ans = short_ans
                remainder = (raw_output[:s] + (leftover if leftover else "") + raw_output[e:]).strip()
            else:
                remainder = (raw_output[:s] + raw_output[e:]).strip()
        else:
            remainder = (raw_output[:s] + raw_output[e:]).strip()

    # 6) Default fallback if still nothing recognized: treat the whole string as remainder
    if ans is None and remainder is None:
        cleaned = raw_output.strip()
        return (cleaned or None, None)

    # 7) Special-case refinement: if remainder is empty and ans exists, try to
    # extract a more specific final answer from inside ans using heuristics
    # (labels/boxed/code), without re-parsing <answer> tags.
    if (remainder is None or remainder == "") and isinstance(ans, str) and ans:
        inner_candidates = []
        # Reuse same heuristics on the inner 'ans' text
        for m in re.finditer(r"```(?:python|py|cpp)\s*([\s\S]*?)```", ans, re.IGNORECASE):
            inner = m.group(1).strip()
            if inner:
                inner_candidates.append((inner, m.span()))
        for m in re.finditer(r"\\boxed\s*", ans):
            p = m.end()
            while p < len(ans) and ans[p].isspace():
                p += 1
            if p < len(ans) and ans[p] == '{':
                depth = 1
                i = p + 1
                close = None
                while i < len(ans):
                    ch = ans[i]
                    prev = ans[i - 1] if i > 0 else ''
                    if ch == '{' and prev != '\\':
                        depth += 1
                    elif ch == '}' and prev != '\\':
                        depth -= 1
                        if depth == 0:
                            close = i
                            break
                    i += 1
                if close is not None:
                    inner = ans[p + 1:close].strip()
                    if inner:
                        inner_candidates.append((inner, (m.start(), close + 1)))
        label_core = r"(?:\*\*\s*)?(?:final\s*answer|answer|decision)(?:\s*\*\*)?\s*:\s*"
        for m in re.finditer(label_core, ans, re.IGNORECASE):
            start = m.end()
            boundary_pat = re.compile(
                r"(?:\n\s*\n|^\s*#{1,6}\s+.*$|^\s*```|^\s*(?:-{3,}|\*{3,}|_{3,})\s*$|^\s*(?:[-*+]\s+|\d+[\.)]\s+)|^\s*(?:Explanation|Reasoning|Analysis|Solution|Notes?|Reference|References|Proof|Derivation|Calculation|Discussion|Conclusion|Summary|설명|해설|풀이|증명|참고|결론|요약)\s*:|\Z)",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            b = boundary_pat.search(ans, pos=start)
            end = b.start() if b else len(ans)
            inner = ans[start:end].strip()
            if inner:
                inner_candidates.append((inner, (m.start(), end)))

        # Also support 'The correct answer is **X**' inside the 'ans' text
        for m in re.finditer(r"the\s+correct\s+answer\s+is\s*:?\s*\*\*(.+?)\*\*", ans, re.IGNORECASE | re.DOTALL):
            inner = (m.group(1) or "").strip()
            if inner:
                inner_candidates.append((inner, m.span()))

        if inner_candidates:
            inner_text, (s, e) = max(inner_candidates, key=lambda x: x[1][1])
            inner_rem = (ans[:s] + ans[e:]).strip()
            ans = inner_text
            remainder = inner_rem or None

    return (remainder or None, ans or None)

def get_inference_results(args):
    print(f"Loading model: {args.model_name}")
    use_api = args.use_proprietary_api

    if use_api:
        sys = "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer with brief explanation. The final answer is enclosed within <answer> </answer> tags, i.e., <answer> final answer here </answer>."
        lower = args.model_name.lower()
        if "claude" in lower:
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY must be set when using Claude models.")
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif "gemini" in lower:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY must be set when using Gemini models.")
            client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set when using OpenAI models.")
            client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        # Set system prompt and special tags on open-source models
        if "deepseek" in args.model_name.lower():
            sys = "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            bos = "<｜begin▁of▁sentence｜>"
            user_desc = "<｜User｜>"
            assis_desc = "<｜Assistant｜><think>\n"
        
        elif "mimo" in args.model_name.lower():
            sys = ""
            bos = "<|im_start|>system\n"
            user_desc = "<|im_end|>\n<|im_start|>user\n"
            assis_desc = "<|im_end|>\n<|im_start|>assistant<think>\n"

        elif "qwen" in args.model_name.lower():
            sys = "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            bos = "<｜begin▁of▁sentence｜>"
            user_desc = "<｜User｜>"
            assis_desc = "<｜Assistant｜><think>\n"

        elif "mistral" in args.model_name.lower():
            sys = '''A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning. Don't mention that this is a summary.

<answer> Then, present a clear final answer to the user. </answer>

Problem:

'''
            bos = "<s>[SYSTEM_PROMPT]"
            user_desc = "[/SYSTEM_PROMPT][INST]"
            assis_desc = "[/INST]<think>\n"

        elif "gpt" in args.model_name.lower():
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

            sys = '''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, final. Channel must be included for every message.<|end|>'''
            dev = '''<|start|>developer<|message|># Instructions
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer with brief explanation. The final answer is enclosed within <answer> </answer> tags, i.e., <answer> final answer here </answer>. Do not explicitly mention the given instructions in your answer.<|end|>'''
            user_desc = "<|start|>user<|message|>"
            assis_desc = "<|start|>assistant<|channel|>analysis<|message|>"
        
        elif "phi" in args.model_name.lower():
            sys = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. The final answer of the Solution should be enclosed within <answer> </answer> tags, i.e., <answer> final answer here </answer>. Now, try to solve the following question through the above guidelines:"
            bos = "<|im_start|>system<|im_sep|>\n"
            user_desc = "<|im_end|>\n   im_start|>user<|im_sep|>"
            assis_desc = "<|im_end|>\n<|im_start|>assistant<|im_sep|><think>\n"
        
        elif "llama" in args.model_name.lower():
            sys = "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            bos = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            user_desc = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            assis_desc = "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|><think>\n"
        else:
            raise ValueError(f"Unsupported open-source model: {args.model_name}")

        # Set sampling parameters and initialize LLM
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32768,
            repetition_penalty=1.0 if "qwen" in args.model_name.lower() else 1.2,
            top_p=1.0,
            top_k=-1
        )

        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.num_gpu,
            gpu_memory_utilization=args.gpu_util,
            trust_remote_code=True,
            **({
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral"
            } if ("mistral" in args.model_name.lower()) else {})
        )

    # Get dataset
    dataset = load_dataset(args)
    results = []

    if hasattr(dataset, "select"):
        ds_len = len(dataset)
        start = max(0, int(args.start_idx))
        end = min(int(args.end_idx), ds_len)
        if end <= start:
            dataset_iter = []
        else:
            dataset_iter = dataset.select(range(start, end))
    else:
        dataset_iter = dataset[args.start_idx:args.end_idx]

    pbar = tqdm(dataset_iter, position=0, leave=True, file=_sys.stdout, disable=False)
    for example in pbar:
        if not args.apply_intervention:
            prefix = "Baseline"
        else:
            prefix = "Intervention"
        pbar.write(
            f"\n\n[{prefix}] [{os.path.basename(args.model_name)}]\n"
            f"[{args.task}] Processing {example['id']}...\n\n"
        )
        parsing_inference_input = import_dataset_utils(example)

        output = ""
        parsed_output = {"reasoning": None, "remainder": None, "answer": None}
        finished = True

        if not use_api:
            # Prepare prompt and get inference results for open-source models
            if "gpt" not in args.model_name.lower():
                if args.apply_intervention:
                    input = _call_parsing_inference_input(parsing_inference_input, example, args) + assis_desc + example["r_prime"]
                else:
                    input = _call_parsing_inference_input(parsing_inference_input, example, args) + assis_desc

                prompt = bos + sys + user_desc + input

                generated_outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
                output = generated_outputs[0].outputs[0].text
                parsed_output = parsing_inference_output("<think>" + output)
                finished = generated_outputs[0].outputs[0].finish_reason == "stop"
            
            elif "gpt" in args.model_name.lower():
                tokenizer = llm.get_tokenizer()

                input = _call_parsing_inference_input(parsing_inference_input, example, args) + "<|end|>"
                if args.apply_intervention:
                    cf_reasoning = assis_desc + example["r_prime"]
                else:
                    cf_reasoning = assis_desc
                
                input = input + "\n" + cf_reasoning
                reasoning_ids = tokenizer(cf_reasoning, add_special_tokens=False)["input_ids"]

                prompt = sys + dev + user_desc + input

                generated_outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
                gen = generated_outputs[0].outputs[0]
                
                if getattr(gen, "finish_reason", None) != "stop":
                    output = gen.text
                    finished = False
                    parsed_output = {"reasoning": output, "remainder": None, "answer": None}
                else:
                    output_tokens = gen.token_ids
                    entries = encoding.parse_messages_from_completion_tokens(reasoning_ids + output_tokens, Role.ASSISTANT)
                    if len(entries) < 2:
                        output = gen.text
                        finished = False
                        parsed_output = {"reasoning": output, "remainder": None, "answer": None}
                    else:
                        generated_reasoning = entries[0].content[0].text
                        if args.apply_intervention:
                            generated_reasoning = generated_reasoning.replace(example["r_prime"], "", 1)
                        output = f"{generated_reasoning}\n</think>\n\n{entries[1].content[0].text}"
                        rem_wo_ans, ans_only = parsing_answer_only(entries[1].content[0].text)
                        parsed_output = {
                            "reasoning": generated_reasoning,
                            "remainder": rem_wo_ans,
                            "answer": ans_only
                        }
                        finished = True
            else:
                raise ValueError(f"Unsupported open-source model: {args.model_name}")
        else:
            user_content = _call_parsing_inference_input(parsing_inference_input, example, args)
            input = user_content
            lower = args.model_name.lower()

            if "claude" in lower:
                if args.apply_intervention:
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": example["r_prime"]},
                        {"role": "user", "content": "Continue the reasoning."}
                    ]
                else:
                    messages = [{"role": "user", "content": user_content}]

                error_message = ""
                response = None
                success = False
                for attempt in range(4):
                    try:
                        response = client.messages.create(
                            model=args.model_name,
                            max_tokens=(64000 if args.task == "paper_review" else 40960),
                            system=sys,
                            thinking={"type": "enabled", "budget_tokens": 32768},
                            messages=messages
                        )
                        success = True
                        break
                    except Exception as e:
                        time.sleep(3)
                        error_message = f"[{attempt+1}TH TRY] {type(e).__name__}: {e}\n\n"

                if not success:
                    output = error_message or "[NO VISIBLE OUTPUT]"
                    parsed_output = {"reasoning": None, "remainder": output, "answer": None}
                    finished = False
                else:
                    reasoning = None
                    answer_text = ""
                    try:
                        for block in response.content:
                            btype = getattr(block, "type", None)
                            if btype == "thinking":
                                reasoning = getattr(block, "thinking", None) or getattr(block, "text", None)
                            elif btype == "text":
                                txt = getattr(block, "text", "")
                                if isinstance(txt, str):
                                    answer_text += txt
                    except Exception:
                        pass
                    if reasoning:
                        output = f"{reasoning}\n</think>\n\n{answer_text}"
                        rem_wo_ans, ans_only = parsing_answer_only(answer_text)
                        parsed_output = {"reasoning": reasoning, "remainder": rem_wo_ans, "answer": ans_only}
                    else:
                        output = answer_text or "[NO VISIBLE OUTPUT]"
                        rem_wo_ans, ans_only = parsing_answer_only(output)
                        parsed_output = {"reasoning": None, "remainder": rem_wo_ans, "answer": ans_only}
                    finished = True

            elif "gemini" in lower:
                if args.apply_intervention:
                    contents = [
                        types.Content(role="user", parts=[{"text": user_content}]),
                        types.Content(role="model", parts=[{"text": example["r_prime"]}]),
                        types.Content(role="user", parts=[{"text": "Continue the reasoning."}])
                    ]
                else:
                    contents = [types.Content(role="user", parts=[{"text": user_content}])]

                error_message = ""
                response = None
                success = False
                for attempt in range(4):
                    try:
                        response = client.models.generate_content(
                            model=args.model_name,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                thinking_config=types.ThinkingConfig(
                                    thinking_budget=32768,
                                    include_thoughts=True
                                )
                            )
                        )
                        success = True
                        break
                    except genai_errors.ServerError as e:
                        time.sleep(3)
                        error_message = f"[{attempt+1}TH TRY] {type(e).__name__}: {e}\n\n"

                if not success:
                    output = error_message or "[NO VISIBLE OUTPUT]"
                    parsed_output = {"reasoning": None, "remainder": output, "answer": None}
                    finished = False
                else:
                    reasoning_parts = []
                    answer_parts = []
                    finish_reason = response.candidates[0].finish_reason if response.candidates else None

                    try:
                        for part in response.candidates[0].content.parts:
                            txt = getattr(part, "text", "")
                            is_thought = getattr(part, "thought", False)
                            if is_thought:
                                if txt is not None:
                                    reasoning_parts.append(txt)
                            else:
                                if txt is not None:
                                    answer_parts.append(txt)
                    except Exception:
                        pass

                    reasoning = "".join(reasoning_parts).strip()
                    answer = "".join(answer_parts).strip()

                    if not answer and finish_reason == "MAX_TOKENS":
                        answer = "[TRUNCATED: Reached MAX_TOKENS during thinking; try lower thinking_budget]"
                        finished = False
                    else:
                        finished = True

                    if reasoning:
                        output = f"{reasoning}\n</think>\n\n{answer}"
                        rem_wo_ans, ans_only = parsing_answer_only(answer)
                        parsed_output = {
                            "reasoning": reasoning,
                            "remainder": rem_wo_ans,
                            "answer": ans_only
                        }
                    else:
                        output = answer or "[NO VISIBLE OUTPUT]"
                        rem_wo_ans, ans_only = parsing_answer_only(output)
                        parsed_output = {
                            "reasoning": None,
                            "remainder": rem_wo_ans,
                            "answer": ans_only
                        }

            else:
                # OpenAI Responses API
                if args.apply_intervention:
                    api_input = [
                        {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
                        {"role": "assistant", "content": [{"type": "output_text", "text": example["r_prime"]}]},
                        {"role": "user", "content": [{"type": "input_text", "text": "Continue the reasoning."}]},
                    ]
                else:
                    api_input = [{"role": "user", "content": [{"type": "input_text", "text": user_content}]}]

                response = client.responses.create(
                    model=args.model_name,
                    max_output_tokens=32768,
                    input=api_input,
                    instructions=sys,
                    reasoning={"summary": "detailed"}
                )

                reasoning_text = None
                final_text = None
                for item in getattr(response, "output", []) or []:
                    parts = getattr(item, "content", None) or []
                    for part in parts:
                        ptype = getattr(part, "type", None)
                        if ptype == "reasoning":
                            rsum = getattr(part, "text", None)
                            if not rsum:
                                rmeta = getattr(part, "reasoning", None)
                                rsum = getattr(rmeta, "summary", None) if rmeta is not None else None
                            reasoning_text = rsum or reasoning_text
                        elif ptype == "output_text":
                            ptxt = getattr(part, "text", None)
                            if ptxt:
                                final_text = (final_text or "") + ptxt

                if not final_text:
                    final_text = getattr(response, "output_text", None) or ""

                if reasoning_text:
                    output = f"{reasoning_text}\n</think>\n\n{final_text}"
                else:
                    output = final_text or "[NO VISIBLE OUTPUT]"

                rem_wo_ans, ans_only = parsing_answer_only(final_text or output)
                parsed_output = {
                    "reasoning": reasoning_text,
                    "remainder": rem_wo_ans,
                    "answer": ans_only
                }
                finished = True
        
        # Ensure finished=false if model returned an empty raw string
        if isinstance(output, str) and output.strip() == "":
            finished = False
        # Ensure finished=false if no reasoning was parsed (open-source only)
        if not use_api:
            try:
                if (parsed_output.get("reasoning") is None):
                    finished = False
                else:
                    r = parsed_output.get("reasoning")
                    if isinstance(r, str) and not re.search(r"[A-Za-z]", r):
                        finished = False
            except Exception:
                # If parsed_output is unexpectedly missing, leave finished as-is
                pass
        # Ensure finished=false if both remainder and answer are None
        try:
            rem = parsed_output.get("remainder")
            ans = parsed_output.get("answer")
            if rem is None and ans is None:
                finished = False
            else:
                rem_has_alpha = isinstance(rem, str) and re.search(r"[A-Za-z]", rem)
                ans_has_alpha = isinstance(ans, str) and re.search(r"[A-Za-z]", ans)
                if (not rem_has_alpha) and (not ans_has_alpha):
                    finished = False
        except Exception:
            pass

        # Append inference results
        results.append({
            "task": args.task,
            "id": example["id"],
            "question": example["question"],
            "options": example["options"],
            "answer": example["answer"],
            "r_prime": example["r_prime"],
            "explanation": example["explanation"],
            "input": input,
            "output": {
                "raw": output,
                "reasoning": parsed_output["reasoning"],
                "remainder": parsed_output["remainder"],
                "answer": parsed_output["answer"],
                "finished": finished,
            }
        })

    # Save inference results
    if not args.apply_intervention:
        out_dir = ROOT / "inference_results" / "baseline" / f"{args.task}"
    else:
        out_dir = ROOT / "inference_results" / "intervened" / f"{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / f"{os.path.basename(args.model_name)}.json"

    existing_inference_results = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f_in:
                existing_inference_results = json.load(f_in)
                if not isinstance(existing_inference_results, list):
                    existing_inference_results = []
        except Exception:
            existing_inference_results = []
    
    merged_results = existing_inference_results + results

    # Simple id-based deduplication (keep last occurrence / overwrite)
    try:
        seen = set()
        deduped_reversed = []
        for it in reversed(merged_results):
            key = it.get("id")
            if key in seen:
                continue
            seen.add(key)
            deduped_reversed.append(it)
        deduped_results = list(reversed(deduped_reversed))
    except Exception:
        deduped_results = merged_results

    with open(output_file, "w") as f_out:
        json.dump(deduped_results, f_out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    
    get_inference_results(args)
