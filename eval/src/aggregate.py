"""Aggregate parsed predictions across all (condition, model, user) runs into _summary CSVs.

Reads:
  eval/runs/{run_id}/config.snapshot.json
  eval/runs/{run_id}/parsed/{user}.json
  eval/runs/{run_id}/responses/{user}.json   (for tokens/cost)

Writes (rebuilt fresh every call):
  eval/runs/_summary/overall__per_user.csv
  eval/runs/_summary/overall__average.csv
  eval/runs/_summary/overall__matrix.csv
  eval/runs/_summary/by_domain__per_user.csv
  eval/runs/_summary/by_domain__average.csv
  eval/runs/_summary/by_interaction__per_user.csv
  eval/runs/_summary/by_interaction__average.csv
  eval/runs/_summary/by_clue_difficulty__per_user.csv
  eval/runs/_summary/by_clue_difficulty__average.csv
  eval/runs/_summary/by_clue_type__per_user.csv
  eval/runs/_summary/by_clue_type__average.csv
  eval/runs/_summary/by_planted__per_user.csv
  eval/runs/_summary/by_planted__average.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import loader

# pandas is imported lazily inside functions so that other CLI commands
# (list-users, dry-run, run) work without pandas installed. Only the
# aggregate step requires pandas.

EVAL_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = EVAL_ROOT / "runs"
SUMMARY_DIR = RUNS_DIR / "_summary"

DOMAIN_SLUGS = {
    "Daily Routine": "daily_routine",
    "Food & Dining": "food_dining",
    "Work & Study": "work_study",
    "Leisure & Entertainment": "leisure",
    "Social & Wellbeing": "social_wellbeing",
    "Shopping & Spending": "shopping",
    "Commuting & Travel": "commuting",
    "Cross-Domain": "cross_domain",
}
INTERACTION_SLUGS = {
    "Search/Recommend": "search_recommend",
    "Consult/Advise": "consult_advise",
    "Action/Execute": "action_execute",
    "Reflect/Comprehend": "reflect_comprehend",
}


def _slug(d: dict[str, str], v: str) -> str:
    return d.get(v, v.lower().replace(" ", "_").replace("&", "and").replace("/", "_"))


def _iter_runs():
    if not RUNS_DIR.exists():
        return
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("_") or run_dir.name.startswith("."):
            continue
        snap = run_dir / "config.snapshot.json"
        if not snap.exists():
            continue
        yield run_dir


def collect_long_records():
    """Return (predictions_df, tokens_df).

    predictions_df rows: one per (condition, model, user, question)
    tokens_df rows:      one per (condition, model, user)
    """
    import pandas as pd  # noqa: F401  (forces ImportError early with helpful msg)

    questions = loader.load_questions()
    survey = loader.load_ground_truth()

    pred_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for run_dir in _iter_runs():
        snap = json.load(open(run_dir / "config.snapshot.json", encoding="utf-8"))
        # Skip non-survey runs (proactive__*, emotion__*) which lack 'condition'
        if "condition" not in snap:
            continue
        condition = snap["condition"]
        model = snap["model"]

        parsed_dir = run_dir / "parsed"
        if not parsed_dir.exists():
            continue

        for parsed_path in sorted(parsed_dir.glob("*.json")):
            user = parsed_path.stem
            preds: dict[str, Any] = json.load(open(parsed_path, encoding="utf-8"))
            gt = (survey.get(user) or {}).get("answers", {}) or {}

            # Tokens / cost
            resp_path = run_dir / "responses" / f"{user}.json"
            pt = ct = 0
            cost = 0.0
            if resp_path.exists():
                try:
                    r = json.load(open(resp_path, encoding="utf-8"))
                    pt = int(r.get("prompt_tokens", 0) or 0)
                    ct = int(r.get("completion_tokens", 0) or 0)
                    cost = float(r.get("est_cost_usd", 0.0) or 0.0)
                except Exception:
                    pass
            token_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "user_slug": user,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "est_cost_usd": cost,
                }
            )

            # Planted info from skeleton.clue_plan
            sk = loader.load_skeleton(user)
            planted_info: dict[str, dict[str, str]] = {}
            if sk:
                for cp in sk.get("clue_plan", []):
                    qid = str(cp.get("question_id"))
                    if qid:
                        planted_info[qid] = {
                            "clue_difficulty": cp.get("clue_difficulty"),
                            "clue_type": cp.get("clue_type"),
                        }

            for qno, q in questions.items():
                pred = preds.get(qno)
                gt_letter = gt.get(qno)
                pi = planted_info.get(qno, {})
                pred_rows.append(
                    {
                        "condition": condition,
                        "model": model,
                        "user_slug": user,
                        "question_no": qno,
                        "domain": q["domain"],
                        "interaction_type": q["interaction_type"],
                        "gt_letter": gt_letter,
                        "pred_letter": pred,
                        "valid_gt": gt_letter is not None,
                        "correct": int(
                            pred is not None
                            and gt_letter is not None
                            and pred == gt_letter
                        ),
                        "planted": qno in planted_info,
                        "clue_difficulty": pi.get("clue_difficulty"),
                        "clue_type": pi.get("clue_type"),
                    }
                )

    return pd.DataFrame(pred_rows), pd.DataFrame(token_rows)


# ---------- helpers for slice CSVs ----------

def _avg_block(per_user_df, value_cols):
    """Macro-mean over users per (condition, model). Adds n_users + accuracy_std for overall."""
    g = per_user_df.groupby(["condition", "model"])
    avg = g[value_cols].mean(numeric_only=True).reset_index()
    avg.insert(2, "n_users", g.size().values)
    return avg


def _write(df, name: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_DIR / name, index=False)


# ---------- per-slice writers ----------

def _overall(pred_df, tokens_df) -> None:
    import pandas as pd  # noqa
    df = pred_df[pred_df["valid_gt"]].copy()
    per_user = (
        df.groupby(["condition", "model", "user_slug"])
        .agg(n_total=("correct", "size"), n_correct=("correct", "sum"))
        .reset_index()
    )
    per_user["accuracy"] = per_user["n_correct"] / per_user["n_total"].clip(lower=1)
    if not tokens_df.empty:
        per_user = per_user.merge(
            tokens_df, on=["condition", "model", "user_slug"], how="left"
        )
    _write(per_user, "overall__per_user.csv")

    g = per_user.groupby(["condition", "model"])
    avg = (
        g.agg(
            n_users=("user_slug", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            est_cost_usd_sum=("est_cost_usd", "sum"),
        )
        .reset_index()
    )
    avg["accuracy_std"] = avg["accuracy_std"].fillna(0.0)
    _write(avg, "overall__average.csv")

    # 2D matrix: rows=condition, cols=model, cell=accuracy_mean
    matrix = avg.pivot(index="condition", columns="model", values="accuracy_mean")
    matrix.to_csv(SUMMARY_DIR / "overall__matrix.csv")


def _by_categorical(
    pred_df,
    cat_col: str,
    slug_map: dict[str, str],
    name: str,
) -> None:
    import pandas as pd  # noqa
    """Per-user × per-category accuracy → wide CSV. Average is mean over users."""
    df = pred_df[pred_df["valid_gt"]].copy()
    df["__slug"] = df[cat_col].apply(lambda v: _slug(slug_map, v))

    per_user = (
        df.groupby(["condition", "model", "user_slug", "__slug"])
        .agg(acc=("correct", "mean"))
        .reset_index()
        .pivot_table(
            index=["condition", "model", "user_slug"],
            columns="__slug",
            values="acc",
        )
        .reset_index()
    )
    per_user.columns.name = None
    _write(per_user, f"{name}__per_user.csv")

    cat_cols = [c for c in per_user.columns if c not in ("condition", "model", "user_slug")]
    avg = _avg_block(per_user, cat_cols)
    _write(avg, f"{name}__average.csv")


def _by_clue_difficulty(pred_df) -> None:
    import pandas as pd  # noqa
    df = pred_df[pred_df["valid_gt"] & pred_df["planted"]].copy()
    if df.empty:
        # still write empty stub so downstream doesn't break
        empty = pd.DataFrame(columns=["condition", "model", "user_slug"])
        _write(empty, "by_clue_difficulty__per_user.csv")
        _write(empty, "by_clue_difficulty__average.csv")
        return

    per_user = (
        df.groupby(["condition", "model", "user_slug", "clue_difficulty"])
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    pivot_acc = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="clue_difficulty",
        values="acc",
    )
    pivot_n = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="clue_difficulty",
        values="n",
        fill_value=0,
    )
    pivot_acc.columns = [f"acc_{c}" for c in pivot_acc.columns]
    pivot_n.columns = [f"n_{c}" for c in pivot_n.columns]
    out = pd.concat([pivot_acc, pivot_n], axis=1).reset_index()
    out.columns.name = None
    _write(out, "by_clue_difficulty__per_user.csv")

    val_cols = [c for c in out.columns if c not in ("condition", "model", "user_slug")]
    avg = _avg_block(out, val_cols)
    _write(avg, "by_clue_difficulty__average.csv")


def _by_clue_type(pred_df) -> None:
    import pandas as pd  # noqa
    df = pred_df[pred_df["valid_gt"] & pred_df["planted"]].copy()
    if df.empty:
        empty = pd.DataFrame(columns=["condition", "model", "user_slug"])
        _write(empty, "by_clue_type__per_user.csv")
        _write(empty, "by_clue_type__average.csv")
        return

    per_user = (
        df.groupby(["condition", "model", "user_slug", "clue_type"])
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    pivot_acc = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="clue_type",
        values="acc",
    )
    pivot_n = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="clue_type",
        values="n",
        fill_value=0,
    )
    pivot_acc.columns = [f"acc_{c}" for c in pivot_acc.columns]
    pivot_n.columns = [f"n_{c}" for c in pivot_n.columns]
    out = pd.concat([pivot_acc, pivot_n], axis=1).reset_index()
    out.columns.name = None
    _write(out, "by_clue_type__per_user.csv")

    val_cols = [c for c in out.columns if c not in ("condition", "model", "user_slug")]
    avg = _avg_block(out, val_cols)
    _write(avg, "by_clue_type__average.csv")


def _by_planted(pred_df) -> None:
    import pandas as pd  # noqa
    df = pred_df[pred_df["valid_gt"]].copy()
    if df.empty:
        empty = pd.DataFrame(columns=["condition", "model", "user_slug"])
        _write(empty, "by_planted__per_user.csv")
        _write(empty, "by_planted__average.csv")
        return

    df["__bucket"] = df["planted"].map({True: "planted", False: "unplanted"})
    per_user = (
        df.groupby(["condition", "model", "user_slug", "__bucket"])
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    pivot_acc = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="__bucket",
        values="acc",
    )
    pivot_n = per_user.pivot_table(
        index=["condition", "model", "user_slug"],
        columns="__bucket",
        values="n",
        fill_value=0,
    )
    pivot_acc.columns = [f"acc_{c}" for c in pivot_acc.columns]
    pivot_n.columns = [f"n_{c}" for c in pivot_n.columns]
    out = pd.concat([pivot_acc, pivot_n], axis=1).reset_index()
    out.columns.name = None
    _write(out, "by_planted__per_user.csv")

    val_cols = [c for c in out.columns if c not in ("condition", "model", "user_slug")]
    avg = _avg_block(out, val_cols)
    _write(avg, "by_planted__average.csv")


# ---------- main ----------

def write_summary() -> Path:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("[aggregate] pandas not installed. Skipping CSV summary.")
        print("            Install with: pip install pandas")
        return SUMMARY_DIR
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pred_df, tokens_df = collect_long_records()
    if pred_df.empty:
        print("[aggregate] No completed runs found under eval/runs/.")
        return SUMMARY_DIR

    _overall(pred_df, tokens_df)
    _by_categorical(pred_df, "domain", DOMAIN_SLUGS, "by_domain")
    _by_categorical(pred_df, "interaction_type", INTERACTION_SLUGS, "by_interaction")
    _by_clue_difficulty(pred_df)
    _by_clue_type(pred_df)
    _by_planted(pred_df)

    print(f"[aggregate] Wrote summary CSVs to {SUMMARY_DIR}")
    return SUMMARY_DIR
