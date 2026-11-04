"""Aggregate proactive-task results across users x models, both TEXT and IMAGE.

Run folder layout (new):
  eval/runs/proactive_text__{user}__{model}/   (one per user/model)
    config.snapshot.json
    targets.json
    parsed/Day{D}.json
    judge/Day{D}.json     (lazily filled)

  eval/runs/proactive_image__{user}__{model}/
    config.snapshot.json
    targets.json
    parsed/{event_id}.json
    judge/{event_id}.json

Outputs (eval/runs/_summary/):
  proactive_text__per_user.csv
  proactive_text__average.csv     # macro-mean across users (per model)
  proactive_text__pool.csv        # micro-mean over all per-anchor predictions (per model)
  proactive_text__report.{md,txt}

  proactive_image__per_user.csv
  proactive_image__pool.csv       # main metric for image (sparse data)
  proactive_image__report.{md,txt}

  proactive__eligibility.csv      # which users qualify for each variant + sample counts
  proactive__eligibility_report.{md,txt}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import llm_judge, loader, render
from .tasks import proactive as ptask

EVAL_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = EVAL_ROOT / "runs"
SUMMARY_DIR = RUNS_DIR / "_summary"


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _iter_runs(prefix: str):
    if not RUNS_DIR.exists():
        return
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            yield d


# ---- Eligibility (combined view) ----------------------------------------

def collect_eligibility() -> list[dict[str, Any]]:
    """One row per user with same-day pool counts.

    n_pos_days     : number of days with >=1 anchor (image + agentHelpReason)
    n_total_anchors: total image-bearing anchors on those days (the sample pool)
    n_pos_anchors  : positives among them
    n_neg_anchors  : negatives among them
    eligible       : n_pos_days >= 1
    """
    all_users = loader.discover_users()
    rows: list[dict[str, Any]] = []
    for u in all_users:
        try:
            traj = loader.load_trajectory(u)
        except Exception:
            continue
        targets = ptask.extract_targets_proactive(traj)
        n_pos_days = len(targets)
        n_total = sum(len(t["anchors"]) for t in targets)
        n_pos = sum(1 for t in targets for a in t["anchors"] if a["is_proactive_gt"])
        n_neg = n_total - n_pos
        rows.append(
            {
                "user_slug": u,
                "n_pos_days": n_pos_days,
                "n_total_anchors": n_total,
                "n_pos_anchors": n_pos,
                "n_neg_anchors": n_neg,
                "eligible": int(n_pos_days >= 1),
            }
        )
    rows.sort(key=lambda r: r["user_slug"])
    return rows


# ---- Reason-judge resolution (shared by text and image) ------------------

def _resolve_judges(
    items: list[tuple[str, str, str]],  # (cache_id, gt_reason, pred_reason)
    judge_dir: Path,
    judge_id_to_filename: dict[str, str],
    judge_model_id: str,
    run_judge: bool,
) -> dict[str, dict[str, Any]]:
    """Run / load judge results for each (cache_id) item.

    judge_id_to_filename maps cache_id -> on-disk filename for grouped storage.
    Returns {cache_id: judge_record}.
    """
    out: dict[str, dict[str, Any]] = {}
    # Group by filename, load existing
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for cid, _, _ in items:
        fname = judge_id_to_filename[cid]
        if fname in grouped:
            continue
        fpath = judge_dir / fname
        if fpath.exists():
            try:
                grouped[fname] = json.load(open(fpath, encoding="utf-8"))
            except Exception:
                grouped[fname] = {}
        else:
            grouped[fname] = {}

    if run_judge:
        for cid, gt_r, pred_r in items:
            fname = judge_id_to_filename[cid]
            cache_dict = grouped[fname]
            if cid in cache_dict and cache_dict[cid].get("match") is not None:
                continue
            rec = llm_judge.judge_reason_match(
                gt_reason=gt_r,
                pred_reason=pred_r,
                judge_model_id=judge_model_id,
            )
            cache_dict[cid] = rec
        # persist
        judge_dir.mkdir(parents=True, exist_ok=True)
        for fname, cache_dict in grouped.items():
            with open(judge_dir / fname, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=2)

    for cid, _, _ in items:
        fname = judge_id_to_filename[cid]
        out[cid] = grouped[fname].get(cid, {})
    return out


# ---- TEXT aggregation ----------------------------------------------------

def collect_text(
    judge_model_id: str = "gpt-5-mini",
    run_judge: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (per_user_rows, per_anchor_rows)."""
    per_user: list[dict[str, Any]] = []
    per_anchor: list[dict[str, Any]] = []

    for run_dir in _iter_runs("proactive_text__"):
        snap_path = run_dir / "config.snapshot.json"
        if not snap_path.exists():
            continue
        snap = json.load(open(snap_path, encoding="utf-8"))
        user_slug = snap.get("user_slug")
        model_key = snap.get("model")
        if not user_slug or not model_key:
            continue

        targets_path = run_dir / "targets.json"
        if not targets_path.exists():
            continue
        targets = json.load(open(targets_path, encoding="utf-8"))

        parsed_dir = run_dir / "parsed"
        judge_dir = run_dir / "judge"

        tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
               "n_anchors": 0, "n_pos_gt": 0}
        judge_items: list[tuple[str, str, str]] = []
        judge_id_to_file: dict[str, str] = {}
        anchor_meta: dict[str, dict[str, Any]] = {}

        for tgt in targets:
            di = tgt["day_index"]
            parsed_path = parsed_dir / f"Day{di}.json"
            parsed_day: dict[str, dict[str, Any]] = {}
            if parsed_path.exists():
                try:
                    parsed_day = json.load(open(parsed_path, encoding="utf-8"))
                except Exception:
                    parsed_day = {}

            mb = ptask.compute_binary_metrics_day(tgt, parsed_day)
            for k in ("tp", "fp", "fn", "tn", "n_anchors", "n_pos_gt"):
                tot[k] += mb[k]

            for a in tgt["anchors"]:
                eid = a["event_id"]
                gt = bool(a["is_proactive_gt"])
                p = parsed_day.get(eid) or {}
                pred = bool(p.get("needs_proactive"))
                anchor_meta[eid] = {
                    "user_slug": user_slug,
                    "model": model_key,
                    "day_index": di,
                    "event_id": eid,
                    "gt": int(gt),
                    "pred": int(pred),
                    "missing": int(bool(p.get("missing"))),
                    "pred_reason": p.get("reason") or "",
                    "gt_reason": a.get("agentHelpReason") or "",
                }
                if gt:
                    judge_items.append((eid, a.get("agentHelpReason") or "",
                                        p.get("reason") or ""))
                    judge_id_to_file[eid] = f"Day{di}.json"

        judge_results = _resolve_judges(
            judge_items, judge_dir, judge_id_to_file,
            judge_model_id=judge_model_id, run_judge=run_judge,
        )

        n_judge_yes = 0
        n_judge_total = tot["n_pos_gt"]
        for eid, _, _ in judge_items:
            if not anchor_meta[eid]["pred"]:
                continue  # missed: pred false -> automatic 0
            jrec = judge_results.get(eid) or {}
            if jrec.get("match"):
                n_judge_yes += 1
                anchor_meta[eid]["judge_match"] = 1
            else:
                anchor_meta[eid]["judge_match"] = 0

        # Emit per-anchor rows
        for eid, m in anchor_meta.items():
            row = dict(m)
            if "judge_match" not in row:
                row["judge_match"] = ""
            per_anchor.append(row)

        accuracy = _safe_div(tot["tp"] + tot["tn"], tot["n_anchors"])
        precision = _safe_div(tot["tp"], tot["tp"] + tot["fp"])
        recall = _safe_div(tot["tp"], tot["tp"] + tot["fn"])
        f1 = _f1(precision, recall)
        rmr = _safe_div(n_judge_yes, n_judge_total)

        per_user.append({
            "user_slug": user_slug,
            "model": model_key,
            "n_days": len(targets),
            "n_anchors": tot["n_anchors"],
            "n_pos_gt": tot["n_pos_gt"],
            "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"], "tn": tot["tn"],
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "reason_match_rate": round(rmr, 4),
            "n_reason_judge": n_judge_total,
        })

    per_user.sort(key=lambda r: (r["model"], r["user_slug"]))
    per_anchor.sort(key=lambda r: (r["model"], r["user_slug"], r["day_index"], r["event_id"]))
    return per_user, per_anchor


# ---- IMAGE aggregation (v3: full binary cls, same layout as text) -------

def collect_image(
    judge_model_id: str = "gpt-5-mini",
    run_judge: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (per_user_rows, per_anchor_rows). Image now has full positive +
    negative samples and per-day grouped storage (parsed/Day{D}.json).
    """
    per_user: list[dict[str, Any]] = []
    per_anchor: list[dict[str, Any]] = []

    for run_dir in _iter_runs("proactive_image__"):
        snap_path = run_dir / "config.snapshot.json"
        if not snap_path.exists():
            continue
        snap = json.load(open(snap_path, encoding="utf-8"))
        user_slug = snap.get("user_slug")
        model_key = snap.get("model")
        if not user_slug or not model_key:
            continue

        targets_path = run_dir / "targets.json"
        if not targets_path.exists():
            continue
        targets = json.load(open(targets_path, encoding="utf-8"))

        parsed_dir = run_dir / "parsed"
        judge_dir = run_dir / "judge"

        tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_anchors": 0, "n_pos_gt": 0}
        judge_items: list[tuple[str, str, str]] = []
        judge_id_to_file: dict[str, str] = {}
        anchor_meta: dict[str, dict[str, Any]] = {}

        for tgt in targets:
            di = tgt["day_index"]
            parsed_path = parsed_dir / f"Day{di}.json"
            parsed_day: dict[str, dict[str, Any]] = {}
            if parsed_path.exists():
                try:
                    parsed_day = json.load(open(parsed_path, encoding="utf-8"))
                except Exception:
                    parsed_day = {}

            mb = ptask.compute_binary_metrics_day(tgt, parsed_day)
            for k in ("tp", "fp", "fn", "tn", "n_anchors", "n_pos_gt"):
                tot[k] += mb[k]

            for a in tgt["anchors"]:
                eid = a["event_id"]
                gt = bool(a["is_proactive_gt"])
                p = parsed_day.get(eid) or {}
                pred = bool(p.get("needs_proactive"))
                anchor_meta[eid] = {
                    "user_slug": user_slug,
                    "model": model_key,
                    "day_index": di,
                    "event_id": eid,
                    "gt": int(gt),
                    "pred": int(pred),
                    "missing": int(bool(p.get("missing"))),
                    "pred_reason": p.get("reason") or "",
                    "gt_reason": a.get("agentHelpReason") or "",
                    "image_relpath": a.get("image_relpath", ""),
                }
                if gt:
                    judge_items.append((eid, a.get("agentHelpReason") or "",
                                        p.get("reason") or ""))
                    judge_id_to_file[eid] = f"Day{di}.json"

        judge_results = _resolve_judges(
            judge_items, judge_dir, judge_id_to_file,
            judge_model_id=judge_model_id, run_judge=run_judge,
        )

        n_judge_yes = 0
        n_judge_total = tot["n_pos_gt"]
        for eid, _, _ in judge_items:
            if not anchor_meta[eid]["pred"]:
                continue
            jrec = judge_results.get(eid) or {}
            if jrec.get("match"):
                n_judge_yes += 1
                anchor_meta[eid]["judge_match"] = 1
            else:
                anchor_meta[eid]["judge_match"] = 0

        for eid, m in anchor_meta.items():
            row = dict(m)
            if "judge_match" not in row:
                row["judge_match"] = ""
            per_anchor.append(row)

        accuracy = _safe_div(tot["tp"] + tot["tn"], tot["n_anchors"])
        precision = _safe_div(tot["tp"], tot["tp"] + tot["fp"])
        recall = _safe_div(tot["tp"], tot["tp"] + tot["fn"])
        f1 = _f1(precision, recall)
        rmr = _safe_div(n_judge_yes, n_judge_total)

        per_user.append({
            "user_slug": user_slug,
            "model": model_key,
            "n_days": len(targets),
            "n_anchors": tot["n_anchors"],
            "n_pos_gt": tot["n_pos_gt"],
            "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"], "tn": tot["tn"],
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "reason_match_rate": round(rmr, 4),
            "n_reason_judge": n_judge_total,
        })

    per_user.sort(key=lambda r: (r["model"], r["user_slug"]))
    per_anchor.sort(key=lambda r: (r["model"], r["user_slug"], r["day_index"], r["event_id"]))
    return per_user, per_anchor


# ---- Pool metrics --------------------------------------------------------

def _pool_text(per_anchor: list[dict[str, Any]], per_user_with_judge: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool (micro) metrics across all anchors per model."""
    by_model: dict[str, dict[str, int]] = {}
    for row in per_anchor:
        m = row["model"]
        d = by_model.setdefault(m, {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                     "n": 0, "n_pos_gt": 0,
                                     "n_judge_yes": 0, "n_judge_total": 0})
        gt = bool(row["gt"])
        pred = bool(row["pred"])
        d["n"] += 1
        if gt and pred: d["tp"] += 1
        elif (not gt) and pred: d["fp"] += 1
        elif gt and (not pred): d["fn"] += 1
        else: d["tn"] += 1
        if gt:
            d["n_pos_gt"] += 1
            d["n_judge_total"] += 1
            if row.get("judge_match") == 1:
                d["n_judge_yes"] += 1

    out: list[dict[str, Any]] = []
    for m, d in sorted(by_model.items()):
        acc = _safe_div(d["tp"] + d["tn"], d["n"])
        prec = _safe_div(d["tp"], d["tp"] + d["fp"])
        rec = _safe_div(d["tp"], d["tp"] + d["fn"])
        f1 = _f1(prec, rec)
        rmr = _safe_div(d["n_judge_yes"], d["n_judge_total"])
        out.append({
            "model": m,
            "n_anchors": d["n"],
            "n_pos_gt": d["n_pos_gt"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "reason_match_rate": round(rmr, 4),
            "n_reason_judge": d["n_judge_total"],
        })
    return out


def _pool_image(per_anchor: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """v3: full binary metrics on image pool (now contains negatives)."""
    by_model: dict[str, dict[str, int]] = {}
    for row in per_anchor:
        m = row["model"]
        d = by_model.setdefault(m, {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                     "n": 0, "n_pos_gt": 0,
                                     "n_judge_yes": 0, "n_judge_total": 0})
        gt = bool(row["gt"])
        pred = bool(row["pred"])
        d["n"] += 1
        if gt and pred: d["tp"] += 1
        elif (not gt) and pred: d["fp"] += 1
        elif gt and (not pred): d["fn"] += 1
        else: d["tn"] += 1
        if gt:
            d["n_pos_gt"] += 1
            d["n_judge_total"] += 1
            if row.get("judge_match") == 1:
                d["n_judge_yes"] += 1

    out: list[dict[str, Any]] = []
    for m, d in sorted(by_model.items()):
        acc = _safe_div(d["tp"] + d["tn"], d["n"])
        prec = _safe_div(d["tp"], d["tp"] + d["fp"])
        rec = _safe_div(d["tp"], d["tp"] + d["fn"])
        f1 = _f1(prec, rec)
        rmr = _safe_div(d["n_judge_yes"], d["n_judge_total"])
        out.append({
            "model": m,
            "n_anchors": d["n"],
            "n_pos_gt": d["n_pos_gt"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "reason_match_rate": round(rmr, 4),
            "n_reason_judge": d["n_judge_total"],
        })
    return out


# ---- Public entry point --------------------------------------------------

def write_summary(*, judge_model_id: str = "gpt-5-mini", run_judge: bool = True) -> Path:
    try:
        import pandas as pd
    except ImportError:
        print("[aggregate-proactive] pandas not installed.")
        return SUMMARY_DIR

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Eligibility (always written) -----------------------------------
    elig_rows = collect_eligibility()
    if elig_rows:
        pd.DataFrame(elig_rows).to_csv(
            SUMMARY_DIR / "proactive__eligibility.csv", index=False
        )
        elig_headers = list(elig_rows[0].keys())
        elig_table = [[r[h] for h in elig_headers] for r in elig_rows]
        n_elig = sum(1 for r in elig_rows if r["eligible"])
        n_pos_days = sum(r["n_pos_days"] for r in elig_rows)
        n_total = sum(r["n_total_anchors"] for r in elig_rows)
        n_pos = sum(r["n_pos_anchors"] for r in elig_rows)
        n_neg = sum(r["n_neg_anchors"] for r in elig_rows)
        elig_summary = [
            ["Eligible users (>=1 pos day)", n_elig, ""],
            ["Total positive days", n_pos_days, ""],
            ["Total anchor samples (image-bearing on pos days)",
             n_total, f"{n_pos} pos / {n_neg} neg"],
        ]
        render.write_report(
            path_md=SUMMARY_DIR / "proactive__eligibility_report.md",
            path_txt=SUMMARY_DIR / "proactive__eligibility_report.txt",
            title="Proactive Task — Eligibility (v3 same-day pool)",
            sections=[
                ("Pool summary",
                 ["metric", "value", "note"],
                 elig_summary),
                ("Per-user breakdown", elig_headers, elig_table),
            ],
        )

    # ---- TEXT -----------------------------------------------------------
    text_user, text_anchor = collect_text(
        judge_model_id=judge_model_id, run_judge=run_judge
    )
    if text_user:
        pd.DataFrame(text_user).to_csv(
            SUMMARY_DIR / "proactive_text__per_user.csv", index=False
        )
        pd.DataFrame(text_anchor).to_csv(
            SUMMARY_DIR / "proactive_text__per_anchor.csv", index=False
        )
        # macro-mean across users (per model)
        df = pd.DataFrame(text_user)
        avg = df.groupby("model").agg(
            n_users=("user_slug", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            reason_match_rate_mean=("reason_match_rate", "mean"),
            reason_match_rate_std=("reason_match_rate", "std"),
        ).reset_index()
        for c in ("accuracy_std", "f1_std", "reason_match_rate_std"):
            avg[c] = avg[c].fillna(0.0)
        for c in avg.columns:
            if avg[c].dtype.kind == "f":
                avg[c] = avg[c].round(4)
        avg.to_csv(SUMMARY_DIR / "proactive_text__average.csv", index=False)

        pool = _pool_text(text_anchor, text_user)
        pd.DataFrame(pool).to_csv(SUMMARY_DIR / "proactive_text__pool.csv", index=False)

        render.write_report(
            path_md=SUMMARY_DIR / "proactive_text__report.md",
            path_txt=SUMMARY_DIR / "proactive_text__report.txt",
            title="Proactive Text — Summary",
            sections=[
                ("Pool metrics (micro across all anchors)",
                 list(pool[0].keys()) if pool else [],
                 [list(r.values()) for r in pool]),
                ("Average across users (macro-mean)",
                 list(avg.columns), avg.values.tolist()),
                ("Per-user breakdown",
                 list(df.columns), df.values.tolist()),
            ],
        )
    else:
        print("[aggregate-proactive] No proactive_text runs found.")

    # ---- IMAGE ----------------------------------------------------------
    img_user, img_anchor = collect_image(
        judge_model_id=judge_model_id, run_judge=run_judge
    )
    if img_user:
        pd.DataFrame(img_user).to_csv(
            SUMMARY_DIR / "proactive_image__per_user.csv", index=False
        )
        pd.DataFrame(img_anchor).to_csv(
            SUMMARY_DIR / "proactive_image__per_anchor.csv", index=False
        )
        # macro-mean across users
        df_img = pd.DataFrame(img_user)
        avg_img = df_img.groupby("model").agg(
            n_users=("user_slug", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            reason_match_rate_mean=("reason_match_rate", "mean"),
            reason_match_rate_std=("reason_match_rate", "std"),
        ).reset_index()
        for c in ("accuracy_std", "f1_std", "reason_match_rate_std"):
            avg_img[c] = avg_img[c].fillna(0.0)
        for c in avg_img.columns:
            if avg_img[c].dtype.kind == "f":
                avg_img[c] = avg_img[c].round(4)
        avg_img.to_csv(SUMMARY_DIR / "proactive_image__average.csv", index=False)

        pool_img = _pool_image(img_anchor)
        pd.DataFrame(pool_img).to_csv(
            SUMMARY_DIR / "proactive_image__pool.csv", index=False
        )

        render.write_report(
            path_md=SUMMARY_DIR / "proactive_image__report.md",
            path_txt=SUMMARY_DIR / "proactive_image__report.txt",
            title="Proactive Image — Summary (v3 binary cls)",
            sections=[
                ("Pool metrics (micro across all anchors)",
                 list(pool_img[0].keys()) if pool_img else [],
                 [list(r.values()) for r in pool_img]),
                ("Average across users (macro-mean)",
                 list(avg_img.columns), avg_img.values.tolist()),
                ("Per-user breakdown",
                 list(df_img.columns), df_img.values.tolist()),
            ],
        )
    else:
        print("[aggregate-proactive] No proactive_image runs found.")

    print(f"[aggregate-proactive] Wrote summary to {SUMMARY_DIR}")
    return SUMMARY_DIR
