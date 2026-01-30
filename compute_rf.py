from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parent


# ----------------------- Filesystem helpers -----------------------
def _load_eval(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ----------------------- Normalizers -----------------------
def _norm_stance(val: str | None) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.replace("’", "'").strip().lower()
    return " ".join(s.split()) if s else None


def _stance_of(node):
    if node is None:
        return None
    if isinstance(node, dict):
        v = node.get("stance")
        return v if (v is None or isinstance(v, str)) else str(v)
    return None


def _bool_of(node):
    if node is None:
        return None
    if isinstance(node, dict):
        b = node.get("identifies_flaw")
        if isinstance(b, bool):
            return b
        if isinstance(b, str):
            return b.strip().lower() in ("true", "1", "yes", "y")
        if isinstance(b, (int, float)):
            return bool(b)
    return None


def _is_valid_eval(item) -> bool:
    ev = item.get("evaluation_result")
    if not isinstance(ev, dict):
        return False
    if not isinstance(ev.get("stance_analysis"), dict):
        return False
    if not isinstance(ev.get("transition_analysis"), dict):
        return False
    out_meta = (item.get("instance") or {}).get("output", {})
    if isinstance(out_meta, dict) and out_meta.get("finished") is False:
        return False
    return True


# ----------------------- Integration logic -----------------------
BOTTOM = "⊥"
PLACEHOLDER = None


def _find_scope_dir(eval_root: Path, synonyms: list[str]) -> Optional[Path]:
    for name in synonyms:
        p = eval_root / name
        if p.exists() and p.is_dir():
            return p
    return None


def _list_task_model_pairs(eval_root: Path) -> list[tuple[str, str]]:
    # Support both main repo naming and test repo naming
    orig_dir = _find_scope_dir(eval_root, ["not_augmented", "baseline", "original"]) or (eval_root / "not_augmented")
    cf_dir = _find_scope_dir(eval_root, ["output_level", "intervened", "counterfactual"]) or (eval_root / "output_level")

    task_to_models: dict[str, set[str]] = {}
    for scope_dir in [orig_dir, cf_dir]:
        if scope_dir is None or (not scope_dir.exists()):
            continue
        for task_dir in sorted([p for p in scope_dir.iterdir() if p.is_dir()]):
            task = task_dir.name
            for fp in task_dir.glob("*.json"):
                model = fp.stem
                task_to_models.setdefault(task, set()).add(model)
    pairs: list[tuple[str, str]] = []
    for task, models in sorted(task_to_models.items()):
        for model in sorted(models):
            pairs.append((task, model))
    return pairs


def _build_integrated_df(eval_root: Path, task: str, model: str) -> pd.DataFrame:
    # Resolve scope dirs with synonyms
    orig_dir = _find_scope_dir(eval_root, ["not_augmented", "baseline", "original"]) or (eval_root / "not_augmented")
    cf_dir = _find_scope_dir(eval_root, ["output_level", "intervened", "counterfactual"]) or (eval_root / "output_level")

    p_orig = orig_dir / task / f"{model}.json"
    p_cf = cf_dir / task / f"{model}.json"
    orig = _load_eval(p_orig)
    cf = _load_eval(p_cf)
    rows: dict[str, dict] = {}

    # Original
    for it in orig:
        if not _is_valid_eval(it):
            continue
        inst = it.get("instance", {}) or {}
        eid = str(inst.get("id")) if inst.get("id") is not None else None
        if not eid:
            continue
        ev = it["evaluation_result"]
        sa = ev.get("stance_analysis") or {}
        ta = ev.get("transition_analysis") or {}
        r = rows.setdefault(eid, {"task": task, "id": eid})

        r["orig_counterfactual_reasoning"] = PLACEHOLDER
        r["orig_model_reasoning"] = _stance_of(sa.get("model_reasoning"))
        r["orig_model_explanation"] = _stance_of(sa.get("model_explanation"))
        r["orig_model_final_answer"] = _stance_of(sa.get("model_final_answer"))

        expl_present = isinstance(sa.get("model_explanation"), dict)
        r["orig_cf_to_mr"] = PLACEHOLDER
        if expl_present:
            r["orig_mr_to_me"] = _bool_of(ta.get("model_reasoning_to_model_explanation"))
            r["orig_me_to_ma"] = _bool_of(ta.get("model_explanation_to_model_final_answer"))
            r["orig_mr_to_ma"] = PLACEHOLDER
        else:
            r["orig_mr_to_me"] = PLACEHOLDER
            r["orig_me_to_ma"] = PLACEHOLDER
            r["orig_mr_to_ma"] = _bool_of(ta.get("model_reasoning_to_model_final_answer"))

    # Counterfactual
    for it in cf:
        if not _is_valid_eval(it):
            continue
        inst = it.get("instance", {}) or {}
        eid = str(inst.get("id")) if inst.get("id") is not None else None
        if not eid:
            continue
        ev = it["evaluation_result"]
        sa = ev.get("stance_analysis") or {}
        ta = ev.get("transition_analysis") or {}
        r = rows.setdefault(eid, {"task": task, "id": eid})

        r["cf_counterfactual_reasoning"] = _stance_of(sa.get("counterfactual_reasoning"))
        r["cf_model_reasoning"] = _stance_of(sa.get("model_subsequent_reasoning"))
        r["cf_model_explanation"] = _stance_of(sa.get("model_explanation"))
        r["cf_model_final_answer"] = _stance_of(sa.get("model_final_answer"))

        expl_present = isinstance(sa.get("model_explanation"), dict)
        r["cf_cf_to_mr"] = _bool_of(ta.get("counterfactual_reasoning_to_model_subsequent_reasoning"))
        if expl_present:
            r["cf_mr_to_me"] = _bool_of(ta.get("model_subsequent_reasoning_to_model_explanation"))
            r["cf_me_to_ma"] = _bool_of(ta.get("model_explanation_to_model_final_answer"))
            r["cf_mr_to_ma"] = PLACEHOLDER
        else:
            r["cf_mr_to_me"] = PLACEHOLDER
            r["cf_me_to_ma"] = PLACEHOLDER
            r["cf_mr_to_ma"] = _bool_of(ta.get("model_subsequent_reasoning_to_model_final_answer")) or _bool_of(
                ta.get("model_reasoning_to_model_final_answer")
            )

    if not rows:
        return pd.DataFrame(columns=["task", "id"])

    df = pd.DataFrame(list(rows.values())).set_index(["task", "id"]).sort_index()

    expected_cols = [
        # original
        "orig_counterfactual_reasoning",
        "orig_model_reasoning",
        "orig_model_explanation",
        "orig_model_final_answer",
        "orig_cf_to_mr",
        "orig_mr_to_me",
        "orig_me_to_ma",
        "orig_mr_to_ma",
        # cf
        "cf_counterfactual_reasoning",
        "cf_model_reasoning",
        "cf_model_explanation",
        "cf_model_final_answer",
        "cf_cf_to_mr",
        "cf_mr_to_me",
        "cf_me_to_ma",
        "cf_mr_to_ma",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Effective stances: original
    def _eff_orig(row):
        S_r = row["orig_model_reasoning"]
        S_e = row["orig_model_explanation"]
        S_a = row["orig_model_final_answer"]
        mr_me = row["orig_mr_to_me"]
        me_ma = row["orig_me_to_ma"]
        mr_ma = row["orig_mr_to_ma"]

        E_r = S_r
        if S_e is None:
            E_e = PLACEHOLDER
        else:
            E_e = S_e if (S_r == S_e or (S_r != S_e and mr_me is True)) else BOTTOM

        if S_e is None:
            if S_r == S_a or (S_r != S_a and mr_ma is True):
                E_a = S_a
            else:
                E_a = BOTTOM
        else:
            if E_e == BOTTOM:
                E_a = BOTTOM
            else:
                E_a = S_a if (E_e == S_a or (E_e != S_a and me_ma is True)) else BOTTOM
        return pd.Series({"E_orig_r": E_r, "E_orig_e": E_e, "E_orig_a": E_a})

    # Effective stances: counterfactual
    def _eff_cf(row):
        S_cf = row["cf_counterfactual_reasoning"]
        S_r = row["cf_model_reasoning"]
        S_e = row["cf_model_explanation"]
        S_a = row["cf_model_final_answer"]
        cf_mr = row["cf_cf_to_mr"]
        mr_me = row["cf_mr_to_me"]
        me_ma = row["cf_me_to_ma"]
        mr_ma = row["cf_mr_to_ma"]

        E_r = S_r if (S_cf == S_r or (S_cf != S_r and cf_mr is True)) else BOTTOM

        if S_e is None:
            E_e = PLACEHOLDER
        else:
            E_e = S_e if (E_r == S_e or (E_r != S_e and mr_me is True)) else BOTTOM

        if S_e is None:
            if E_r == S_a or (E_r != S_a and mr_ma is True):
                E_a = S_a
            else:
                E_a = BOTTOM
        else:
            if E_e == BOTTOM:
                E_a = BOTTOM
            else:
                E_a = S_a if (E_e == S_a or (E_e != S_a and me_ma is True)) else BOTTOM
        return pd.Series({"E_cf_r": E_r, "E_cf_e": E_e, "E_cf_a": E_a})

    eff_orig = df.apply(_eff_orig, axis=1)
    eff_cf = df.apply(_eff_cf, axis=1)
    out = pd.concat([df, eff_orig, eff_cf], axis=1)

    out["chi_o"] = out["E_orig_a"].apply(lambda x: False if (x is None or x == BOTTOM) else True)
    out["chi_o'"] = out["E_cf_a"].apply(lambda x: False if (x is None or x == BOTTOM) else True)

    def _kappa(row):
        ea_o, ea_op = row.get("E_orig_a"), row.get("E_cf_a")
        final_changed = (ea_o != ea_op) and (ea_o is not None) and (ea_op is not None)
        S_r_orig = _norm_stance(row.get("orig_model_reasoning"))
        S_r_cf = _norm_stance(row.get("cf_model_reasoning"))
        clause2 = (S_r_orig is not None) and (S_r_cf is not None) and (S_r_orig != S_r_cf)
        return bool(final_changed or clause2)

    out["kappa(o,o')"] = out.apply(_kappa, axis=1)
    out["RF_indicator"] = out["chi_o"] & out["chi_o'"] & out["kappa(o,o')"]
    out = out.reset_index()
    out["model"] = model
    return out


def _build_all_df(eval_root: Path) -> pd.DataFrame:
    pairs = _list_task_model_pairs(eval_root)
    frames: list[pd.DataFrame] = []
    for task, model in pairs:
        try:
            df = _build_integrated_df(eval_root, task, model)
        except Exception as e:
            # continue on malformed files
            df = pd.DataFrame(columns=["task", "id", "model"])
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["task", "id", "model"])
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values(["task", "model", "id"], kind="stable").reset_index(drop=True)


def _filtered_df(df: pd.DataFrame, task: Optional[str], model_name: Optional[str], apply_global_exclusion: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["task", "id", "model"])
    mask = pd.Series(True, index=df.index)
    if task is not None:
        mask &= (df["task"] == task)
    if model_name is not None:
        # Evaluation outputs are saved with os.path.basename(model_name)
        model_base = Path(model_name).name
        mask &= (df["model"] == model_name) | (df["model"] == model_base)
    if apply_global_exclusion:
        try:
            mask &= ~((df["task"] == "paper_review") & (df["model"] == "Qwen3-8B"))
        except Exception:
            pass
    return df.loc[mask].copy()


def sanitize_token(s: Optional[str]) -> str:
    if not s:
        return "ALL"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def compute_rf_and_coverage(evaluator_name: str, task: Optional[str], model_name: Optional[str]):
    eval_root = ROOT / "evaluation_results" / evaluator_name
    df_all = _build_all_df(eval_root)
    base = _filtered_df(df_all, task=task, model_name=model_name, apply_global_exclusion=True)

    # Contrast mask: normalized orig_model_reasoning vs cf_counterfactual_reasoning differ
    if base.empty:
        n_total = 0
        n_contrast = 0
        coverage = None
        rf_score = None
    else:
        r = base["orig_model_reasoning"].apply(_norm_stance) if "orig_model_reasoning" in base.columns else pd.Series([None] * len(base))
        c = base["cf_counterfactual_reasoning"].apply(_norm_stance) if "cf_counterfactual_reasoning" in base.columns else pd.Series([None] * len(base))
        m_contrast = r.notna() & c.notna() & (r != c)
        n_total = int(len(base))
        n_contrast = int(m_contrast.sum())
        coverage = float(n_contrast / n_total) if n_total > 0 else None

        rf_score = None
        if n_contrast > 0 and "RF_indicator" in base.columns:
            sub = base.loc[m_contrast]
            rf_score = float(sub["RF_indicator"].astype(float).mean()) if len(sub) else None

    return {
        "evaluator_name": evaluator_name,
        "task": task or "ALL",
        "model": model_name or "ALL",
        "n_total": n_total,
        "n_contrast": n_contrast,
        "coverage": coverage,
        "rf_score": rf_score,
        "eval_root": str(eval_root),
    }


def main():
    parser = argparse.ArgumentParser(description=(
        "Aggregate RF score and contrast coverage from evaluation_results/{evaluator_name}.\n"
        "- When neither --model-name nor --task is given: aggregates across all models and tasks.\n"
        "- When only --model-name is given: aggregates across all tasks for that model.\n"
        "- When only --task is given: aggregates across all models for that task."
    ))
    parser.add_argument("--evaluator_name", type=str, default="o3", help="Evaluator subdirectory under evaluation_results (default: o3)")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model name to filter.")
    parser.add_argument("--task", type=str, default=None, help="Optional task name to filter.")
    parser.add_argument("--outfile", type=str, default=None, help="Optional filename for CSV output. Defaults to rf_summary*.csv in evaluator dir.")
    args = parser.parse_args()

    res = compute_rf_and_coverage(evaluator_name=args.evaluator_name, task=args.task, model_name=args.model_name)

    out_dir = Path(res["eval_root"])  # type: ignore[index]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.outfile:
        out_path = out_dir / args.outfile
    else:
        suffix = []
        if args.model_name:
            suffix.append(f"model_{sanitize_token(args.model_name)}")
        if args.task:
            suffix.append(f"task_{sanitize_token(args.task)}")
        name = "rf_summary" + ("_" + "_".join(suffix) if suffix else "_overall") + ".csv"
        out_path = out_dir / name

    fieldnames = ["evaluator_name", "task", "model", "n_total", "n_contrast", "coverage", "rf_score"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({k: res.get(k) for k in fieldnames})

    print(f"Saved summary to: {out_path}")
    print({k: res.get(k) for k in fieldnames})


if __name__ == "__main__":
    main()
