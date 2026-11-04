"""Eval CLI.

Usage examples:
  python -m eval.src.cli list-users
  python -m eval.src.cli list-config
  python -m eval.src.cli eligibility
  python -m eval.src.cli run [--users ...] [--conditions ...] [--models ...]   # survey
  python -m eval.src.cli run --task proactive_text  --users ... --models ...
  python -m eval.src.cli run --task proactive_image --users ... --models ...
  python -m eval.src.cli run --task emotion_text    --users ... --models ...
  python -m eval.src.cli run --task emotion_image   --users ... --models ...
  python -m eval.src.cli aggregate [--task survey|proactive|emotion|all]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_PRINT_LOCK = threading.Lock()


def _safe_print(msg: str, *, file=None) -> None:
    """Thread-safe print so concurrent workers don't tear lines."""
    with _PRINT_LOCK:
        print(msg, file=file)
        if file is not None:
            file.flush()
        else:
            sys.stdout.flush()


def _parallel_workers() -> int:
    return max(1, int(os.environ.get("EVAL_PARALLEL", "1")))

import yaml

from . import (
    ablation as _ablation,
    aggregate,
    aggregate_emotion,
    aggregate_proactive,
    llm_client,
    loader,
    parser as resp_parser,
    prompt_builder,
)
from .tasks import emotion as etask
from .tasks import proactive as ptask

EVAL_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = EVAL_ROOT / "runs"
CONDITIONS_YAML = EVAL_ROOT / "configs" / "conditions.yaml"
MODELS_YAML = EVAL_ROOT / "configs" / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    conds = _load_yaml(CONDITIONS_YAML)["conditions"]
    models = _load_yaml(MODELS_YAML)["models"]
    return conds, models


def _estimate_cost(prompt_tokens: int, completion_tokens: int, model_cfg: dict[str, Any]) -> float:
    p = prompt_tokens / 1_000_000.0 * float(model_cfg.get("cost_per_1m_prompt_usd", 0) or 0)
    c = completion_tokens / 1_000_000.0 * float(model_cfg.get("cost_per_1m_completion_usd", 0) or 0)
    return round(p + c, 6)


# ===== Survey runner (unchanged) ==========================================

def _save_run_outputs(run_dir, user_slug, record, parsed, snapshot):
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "responses" / f"{user_slug}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(run_dir / "parsed" / f"{user_slug}.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    snap_path = run_dir / "config.snapshot.json"
    if not snap_path.exists():
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)


def run_one(
    user_slug, condition_key, model_key, conds, models, persona_data, questions,
    *, dry_run=False, ablation_id: str | None = None,
):
    cond_cfg = conds[condition_key]
    model_cfg = models[model_key]
    ev_filter = _ablation.build_event_filter(ablation_id) if ablation_id else None
    system, user_prompt = prompt_builder.build_prompt(
        user_slug=user_slug, cond_cfg=cond_cfg,
        questions=questions, persona_data=persona_data,
        event_domain_filter=ev_filter,
    )
    if dry_run:
        return {"user_slug": user_slug, "condition": condition_key, "model": model_key,
                "system_chars": len(system), "user_chars": len(user_prompt),
                "approx_user_tokens": len(user_prompt) // 4}

    record = llm_client.call_llm(
        model_id=model_cfg["model_id"], system=system, user=user_prompt,
        temperature=float(model_cfg.get("temperature", 0.0)),
        max_tokens=int(model_cfg.get("max_tokens", 2000)),
        provider=model_cfg.get("provider", "openai"),
        reasoning=model_cfg.get("reasoning"),
        provider_routing=model_cfg.get("provider_routing"),
        project_id=model_cfg.get("project_id"),
        location=model_cfg.get("location"),
        thinking_budget=model_cfg.get("thinking_budget"),
    )
    record["est_cost_usd"] = _estimate_cost(
        record.get("prompt_tokens", 0) or 0,
        record.get("completion_tokens", 0) or 0, model_cfg,
    )
    parsed = resp_parser.parse_letters(record["content"], list(questions.keys()))
    n_missing = sum(1 for v in parsed.values() if v is None)
    if ablation_id:
        run_id = f"{condition_key}__{model_key}__{ablation_id}"
    else:
        run_id = f"{condition_key}__{model_key}"
    run_dir = RUNS_DIR / run_id
    snapshot = {
        "run_id": run_id, "condition": condition_key, "model": model_key,
        "model_id": model_cfg["model_id"],
        "ablation": ablation_id,
        "temperature": model_cfg.get("temperature"),
        "max_tokens": model_cfg.get("max_tokens"),
        "condition_cfg": cond_cfg,
        "cost_per_1m_prompt_usd": model_cfg.get("cost_per_1m_prompt_usd"),
        "cost_per_1m_completion_usd": model_cfg.get("cost_per_1m_completion_usd"),
    }
    _save_run_outputs(run_dir, user_slug, record, parsed, snapshot)
    return {"user_slug": user_slug, "condition": condition_key, "model": model_key,
            "ablation": ablation_id,
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens"),
            "est_cost_usd": record.get("est_cost_usd"),
            "cached": record.get("cached", False), "n_missing": n_missing}


# ===== Proactive runners ==================================================

def run_one_proactive_text(user_slug, model_key, models, *, dry_run=False):
    model_cfg = models[model_key]
    traj = loader.load_trajectory(user_slug)
    targets = ptask.extract_targets_text(traj, k=5)
    if not targets:
        return {"user_slug": user_slug, "model": model_key, "skipped": True,
                "reason": "no anchor events in last 5 days", "n_days": 0}

    run_id = f"proactive_text__{user_slug}__{model_key}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "targets.json", "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    snap_path = run_dir / "config.snapshot.json"
    if not snap_path.exists():
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id, "task": "proactive_text",
                "user_slug": user_slug, "model": model_key,
                "model_id": model_cfg["model_id"],
                "temperature": model_cfg.get("temperature"),
                "max_tokens": model_cfg.get("max_tokens"),
                "n_target_days": len(targets),
            }, f, ensure_ascii=False, indent=2)

    summary = {"user_slug": user_slug, "model": model_key, "n_days": len(targets),
               "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0,
               "cached_units": 0, "skipped": False}

    for tgt in targets:
        di = tgt["day_index"]
        anchor_ids = [a["event_id"] for a in tgt["anchors"]]
        sys_p, user_p = ptask.build_prompt_text(traj, tgt)
        if dry_run:
            print(f"  Day{di}: anchors={len(anchor_ids)} chars={len(user_p)}")
            continue
        record = llm_client.call_llm(
            model_id=model_cfg["model_id"], system=sys_p, user=user_p,
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(model_cfg.get("max_tokens", 16000)),
            provider=model_cfg.get("provider", "openai"),
            reasoning=model_cfg.get("reasoning"),
            provider_routing=model_cfg.get("provider_routing"),
            project_id=model_cfg.get("project_id"),
            location=model_cfg.get("location"),
            thinking_budget=model_cfg.get("thinking_budget"),
        )
        record["est_cost_usd"] = _estimate_cost(
            record.get("prompt_tokens", 0) or 0,
            record.get("completion_tokens", 0) or 0, model_cfg,
        )
        parsed = ptask.parse_response_text(record["content"], anchor_ids)
        with open(run_dir / "responses" / f"Day{di}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(run_dir / "parsed" / f"Day{di}.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        summary["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
        summary["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
        summary["est_cost_usd"] += float(record.get("est_cost_usd", 0.0) or 0.0)
        if record.get("cached"):
            summary["cached_units"] += 1
    return summary


def run_one_proactive_image(user_slug, model_key, models, *, dry_run=False):
    """v3: same per-day pool as text; output is multi-event JSON keyed by event_id.
    One LLM call per positive day, with N images attached (one per anchor).
    """
    model_cfg = models[model_key]
    traj = loader.load_trajectory(user_slug)
    targets = ptask.extract_targets_proactive(traj)
    if not targets:
        return {"user_slug": user_slug, "model": model_key, "skipped": True,
                "reason": "no positive days (anchor + image + agentHelpReason)",
                "n_days": 0}

    run_id = f"proactive_image__{user_slug}__{model_key}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "targets.json", "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    snap_path = run_dir / "config.snapshot.json"
    if not snap_path.exists():
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id, "task": "proactive_image",
                "user_slug": user_slug, "model": model_key,
                "model_id": model_cfg["model_id"],
                "temperature": model_cfg.get("temperature"),
                "max_tokens": model_cfg.get("max_tokens"),
                "n_target_days": len(targets),
                "n_target_anchors": sum(len(t["anchors"]) for t in targets),
            }, f, ensure_ascii=False, indent=2)

    summary = {"user_slug": user_slug, "model": model_key,
               "n_days": len(targets),
               "n_anchors": sum(len(t["anchors"]) for t in targets),
               "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0,
               "cached_units": 0, "skipped": False}

    for tgt in targets:
        di = tgt["day_index"]
        anchor_ids = [a["event_id"] for a in tgt["anchors"]]
        sys_p, user_p, image_paths = ptask.build_prompt_image(traj, tgt)
        if dry_run:
            print(f"  Day{di}: anchors={len(anchor_ids)} images={len(image_paths)} chars={len(user_p)}")
            continue
        record = llm_client.call_llm(
            model_id=model_cfg["model_id"], system=sys_p, user=user_p,
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(model_cfg.get("max_tokens", 16000)),
            image_paths=image_paths,
            provider=model_cfg.get("provider", "openai"),
            reasoning=model_cfg.get("reasoning"),
            provider_routing=model_cfg.get("provider_routing"),
            project_id=model_cfg.get("project_id"),
            location=model_cfg.get("location"),
            thinking_budget=model_cfg.get("thinking_budget"),
        )
        record["est_cost_usd"] = _estimate_cost(
            record.get("prompt_tokens", 0) or 0,
            record.get("completion_tokens", 0) or 0, model_cfg,
        )
        parsed = ptask.parse_response_image(record["content"], anchor_ids)
        with open(run_dir / "responses" / f"Day{di}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(run_dir / "parsed" / f"Day{di}.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        summary["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
        summary["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
        summary["est_cost_usd"] += float(record.get("est_cost_usd", 0.0) or 0.0)
        if record.get("cached"):
            summary["cached_units"] += 1
    return summary


# ===== Emotion runners ====================================================

def _run_emotion_generic(user_slug, model_key, models, variant, *, dry_run=False):
    """variant: 'text' or 'image'."""
    model_cfg = models[model_key]
    traj = loader.load_trajectory(user_slug)
    if variant == "text":
        targets = etask.extract_targets_text(traj, k=5)
    else:
        targets = etask.extract_targets_image(traj)
    if not targets:
        return {"user_slug": user_slug, "model": model_key, "skipped": True,
                "reason": f"no targets for emotion_{variant}", "n_events": 0}

    run_id = f"emotion_{variant}__{user_slug}__{model_key}"
    run_dir = RUNS_DIR / run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "targets.json", "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    snap_path = run_dir / "config.snapshot.json"
    if not snap_path.exists():
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id, "task": f"emotion_{variant}",
                "user_slug": user_slug, "model": model_key,
                "model_id": model_cfg["model_id"],
                "temperature": model_cfg.get("temperature"),
                "max_tokens": model_cfg.get("max_tokens"),
                "n_target_events": len(targets),
            }, f, ensure_ascii=False, indent=2)

    summary = {"user_slug": user_slug, "model": model_key, "n_events": len(targets),
               "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0,
               "cached_units": 0, "skipped": False}

    for tgt in targets:
        eid = tgt["event_id"]
        if variant == "text":
            sys_p, user_p = etask.build_prompt_text(traj, tgt)
            image_paths = None
        else:
            sys_p, user_p, image_paths = etask.build_prompt_image(traj, tgt)
        if dry_run:
            print(f"  {eid}: chars={len(user_p)} imgs={len(image_paths or [])}")
            continue
        record = llm_client.call_llm(
            model_id=model_cfg["model_id"], system=sys_p, user=user_p,
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(model_cfg.get("max_tokens", 16000)),
            image_paths=image_paths,
            provider=model_cfg.get("provider", "openai"),
            reasoning=model_cfg.get("reasoning"),
            provider_routing=model_cfg.get("provider_routing"),
            project_id=model_cfg.get("project_id"),
            location=model_cfg.get("location"),
            thinking_budget=model_cfg.get("thinking_budget"),
        )
        record["est_cost_usd"] = _estimate_cost(
            record.get("prompt_tokens", 0) or 0,
            record.get("completion_tokens", 0) or 0, model_cfg,
        )
        parsed = etask.parse_response(record["content"])
        with open(run_dir / "responses" / f"{eid}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(run_dir / "parsed" / f"{eid}.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        summary["prompt_tokens"] += int(record.get("prompt_tokens", 0) or 0)
        summary["completion_tokens"] += int(record.get("completion_tokens", 0) or 0)
        summary["est_cost_usd"] += float(record.get("est_cost_usd", 0.0) or 0.0)
        if record.get("cached"):
            summary["cached_units"] += 1
    return summary


def run_one_emotion_text(user_slug, model_key, models, *, dry_run=False):
    return _run_emotion_generic(user_slug, model_key, models, "text", dry_run=dry_run)


def run_one_emotion_image(user_slug, model_key, models, *, dry_run=False):
    return _run_emotion_generic(user_slug, model_key, models, "image", dry_run=dry_run)


# ===== Subcommands ========================================================

def cmd_list_users(_args):
    for u in loader.discover_users():
        print(u)


def cmd_list_config(_args):
    conds, models = load_config()
    print("Conditions:")
    for k, v in conds.items():
        print(f"  - {k}: {v.get('description','')}")
    print("\nModels:")
    for k, v in models.items():
        print(f"  - {k}: model_id={v.get('model_id')}")


def cmd_eligibility(_args):
    """Print and write eligibility tables for proactive + emotion (v3 unified pools)."""
    p_rows = aggregate_proactive.collect_eligibility()
    e_rows = aggregate_emotion.collect_eligibility()

    print("=" * 72)
    print("PROACTIVE eligibility (same-day pool: anchor+image on positive days)")
    print("=" * 72)
    print(f"{'user':<8} {'pos_days':>9} {'total':>6} {'pos':>5} {'neg':>5} {'OK':>4}")
    for r in p_rows:
        print(f"{r['user_slug']:<8} {r['n_pos_days']:>9} "
              f"{r['n_total_anchors']:>6} {r['n_pos_anchors']:>5} "
              f"{r['n_neg_anchors']:>5} {r['eligible']:>4}")
    p_elig = sum(1 for r in p_rows if r["eligible"])
    p_total = sum(r["n_total_anchors"] for r in p_rows)
    p_pos = sum(r["n_pos_anchors"] for r in p_rows)
    p_neg = sum(r["n_neg_anchors"] for r in p_rows)
    print(f"\nTotals: eligible users={p_elig}, samples={p_total} "
          f"({p_pos} pos / {p_neg} neg)")

    print("\n" + "=" * 72)
    print("EMOTION eligibility (last 5 days + image + mood)")
    print("=" * 72)
    print(f"{'user':<8} {'samples':>8} {'pos':>4} {'neg':>4} {'mix':>4} {'OK':>4}")
    for r in e_rows:
        print(f"{r['user_slug']:<8} {r['n_samples']:>8} "
              f"{r['n_pos']:>4} {r['n_neg']:>4} {r['n_mix']:>4} "
              f"{r['eligible']:>4}")
    e_elig = sum(1 for r in e_rows if r["eligible"])
    e_total = sum(r["n_samples"] for r in e_rows)
    e_pos = sum(r["n_pos"] for r in e_rows)
    e_neg = sum(r["n_neg"] for r in e_rows)
    e_mix = sum(r["n_mix"] for r in e_rows)
    print(f"\nTotals: eligible users={e_elig}, samples={e_total} "
          f"({e_pos} pos / {e_neg} neg / {e_mix} mixed)")

    aggregate_proactive.write_summary(run_judge=False)
    aggregate_emotion.write_summary()


def cmd_publish(_args):
    """Organize _summary/ flat outputs into _results/{task}/N{n_users}_M{n_models}/.

    Inspects existing run directories to discover (task, user, model) tuples,
    then copies the matching summary CSV/MD/TXT files into a structured tree
    with consistent file names (no task prefix). Use this after a run to get a
    clean, comparable snapshot per task.
    """
    import shutil
    SUMMARY = RUNS_DIR / "_summary"
    RESULTS = RUNS_DIR / "_results"

    if not SUMMARY.exists():
        print("No _summary/ found — run something first."); return

    # ----- task_setting -> {file_in_summary: file_in_results} mapping -----
    PROX_EMO_FILES = {
        "pool.csv":         "pool.csv",
        "average.csv":      "average.csv",
        "per_user.csv":     "per_user.csv",
        "per_anchor.csv":   "per_anchor.csv",
        "per_event.csv":    "per_event.csv",
        "per_sample.csv":   "per_sample.csv",
        "report.md":        "report.md",
        "report.txt":       "report.txt",
    }
    task_settings = ["proactive_text", "proactive_image",
                     "emotion_text",   "emotion_image"]

    for ts in task_settings:
        run_dirs = sorted(
            d for d in RUNS_DIR.iterdir()
            if d.is_dir() and d.name.startswith(f"{ts}__")
        )
        if not run_dirs:
            continue
        users, models = set(), set()
        for d in run_dirs:
            tail = d.name[len(ts) + 2 :]      # strip "{ts}__"
            if "__" in tail:
                u, m = tail.split("__", 1)
                users.add(u); models.add(m)
        n_u = len(users); n_m = len(models)
        target = RESULTS / ts / f"N{n_u}_M{n_m}"
        target.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src_name, dst_name in PROX_EMO_FILES.items():
            src = SUMMARY / f"{ts}__{src_name}"
            if src.exists():
                shutil.copy(src, target / dst_name)
                copied += 1
        # eligibility (shared across the two settings of one task)
        prefix = "proactive" if ts.startswith("proactive") else "emotion"
        for ext in ("csv", "md", "txt"):
            elig_src = SUMMARY / f"{prefix}__eligibility.{ext}" if ext == "csv" \
                else SUMMARY / f"{prefix}__eligibility_report.{ext}"
            if elig_src.exists():
                shutil.copy(elig_src, target / elig_src.name.replace(f"{prefix}__", ""))

        # meta.json (records what was published)
        meta = {
            "task_setting": ts,
            "n_users": n_u,
            "n_models": n_m,
            "users": sorted(users),
            "models": sorted(models),
            "files_published": copied,
        }
        with open(target / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[publish] {ts}: {n_u} users x {n_m} models -> {target}")

    # ----- Survey (different structure: condition x model x user) -----
    survey_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and "__" in d.name
        and not d.name.startswith(("proactive_", "emotion_"))
        and not d.name.startswith("_")
    )
    if survey_dirs:
        users_s, models_s, conds_s = set(), set(), set()
        for d in survey_dirs:
            parts = d.name.split("__")
            if len(parts) == 2:
                conds_s.add(parts[0]); models_s.add(parts[1])
        # Read user list from any per_user CSV
        ov_pu = SUMMARY / "overall__per_user.csv"
        if ov_pu.exists():
            try:
                import pandas as pd
                users_s = set(pd.read_csv(ov_pu)["user_slug"].unique())
            except Exception:
                pass
        n_u = len(users_s); n_m = len(models_s); n_c = len(conds_s)
        target = RESULTS / "survey" / f"N{n_u}_M{n_m}_C{n_c}"
        target.mkdir(parents=True, exist_ok=True)

        survey_files = [
            "overall__average.csv", "overall__matrix.csv", "overall__per_user.csv",
            "by_domain__average.csv", "by_domain__per_user.csv",
            "by_interaction__average.csv", "by_interaction__per_user.csv",
            "by_clue_difficulty__average.csv", "by_clue_difficulty__per_user.csv",
            "by_clue_type__average.csv", "by_clue_type__per_user.csv",
            "by_planted__average.csv", "by_planted__per_user.csv",
        ]
        copied = 0
        for fname in survey_files:
            src = SUMMARY / fname
            if src.exists():
                shutil.copy(src, target / fname.replace("__", "_"))
                copied += 1
        meta = {"task": "survey", "n_users": n_u, "n_models": n_m,
                "n_conditions": n_c, "users": sorted(users_s),
                "models": sorted(models_s), "conditions": sorted(conds_s),
                "files_published": copied}
        with open(target / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[publish] survey: {n_u} users x {n_m} models x {n_c} conds -> {target}")

    print(f"\nResults tree: {RESULTS}")


def cmd_aggregate(args):
    task = getattr(args, "task", "survey")
    if task == "survey":
        aggregate.write_summary()
    elif task == "proactive":
        aggregate_proactive.write_summary(
            judge_model_id=args.judge_model_id, run_judge=not args.no_judge,
        )
    elif task == "emotion":
        aggregate_emotion.write_summary()
    elif task == "all":
        aggregate.write_summary()
        aggregate_proactive.write_summary(
            judge_model_id=args.judge_model_id, run_judge=not args.no_judge,
        )
        aggregate_emotion.write_summary()
    else:
        print(f"Unknown task: {task}", file=sys.stderr); sys.exit(2)


# ---- Driver for proactive/emotion variants ------------------------------

_TASK_RUNNER = {
    "proactive_text": (run_one_proactive_text, ptask.discover_eligible_users_proactive),
    "proactive_image": (run_one_proactive_image, ptask.discover_eligible_users_proactive),
    "emotion_text": (run_one_emotion_text, etask.discover_eligible_users_emotion),
    "emotion_image": (run_one_emotion_image, etask.discover_eligible_users_emotion),
}


def _run_task_variant(args, task: str):
    _, models = load_config()
    runner, eligible_fn = _TASK_RUNNER[task]
    available = set(loader.discover_users())
    eligible_set = set(eligible_fn())

    requested = args.users or sorted(eligible_set)
    requested = [u for u in requested if u in available]
    skipped_ineligible = [u for u in requested if u not in eligible_set]
    users = [u for u in requested if u in eligible_set]

    if skipped_ineligible:
        print(f"Skipping ineligible users for {task}: {skipped_ineligible}")
    if not users:
        print(f"No eligible users for {task}. Aborting.")
        return

    model_keys = args.models or list(models.keys())
    bad_models = [m for m in model_keys if m not in models]
    if bad_models:
        print(f"ERROR: unknown models: {bad_models}. Known: {list(models.keys())}")
        sys.exit(2)

    print(f"Task:       {task}")
    print(f"Users:      {users}  (eligible: {len(eligible_set)})")
    print(f"Models:     {model_keys}")
    print(f"Total runs: {len(users) * len(model_keys)}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 60)

    total = len(users) * len(model_keys)
    work = [(u, mk) for mk in model_keys for u in users]
    workers = _parallel_workers()
    print(f"Parallel:   {workers} worker(s)")

    n_ok = n_err = 0
    total_cost = 0.0
    counter = {"i": 0}

    def _do(u: str, mk: str):
        return runner(user_slug=u, model_key=mk, models=models, dry_run=args.dry_run)

    if workers == 1:
        for u, mk in work:
            counter["i"] += 1
            tag = f"[{counter['i']}/{total}] {u} | {mk}"
            try:
                info = _do(u, mk)
                n_ok += 1
                if info.get("skipped"):
                    _safe_print(f"{tag} SKIP: {info.get('reason')}"); continue
                cost = info.get("est_cost_usd") or 0.0
                total_cost += cost
                if args.dry_run:
                    _safe_print(f"{tag} DRY  n={info.get('n_days') or info.get('n_events') or info.get('n_samples')}")
                else:
                    n_units = info.get("n_days") or info.get("n_events") or info.get("n_samples")
                    cached = info.get("cached_units") or 0
                    _safe_print(
                        f"{tag} pt={info['prompt_tokens']} ct={info['completion_tokens']} "
                        f"cost=${cost:.4f} units={n_units} cached={cached}"
                    )
            except Exception as e:
                n_err += 1
                _safe_print(f"{tag} ERROR: {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_do, u, mk): (u, mk) for u, mk in work}
            for fut in as_completed(futs):
                u, mk = futs[fut]
                counter["i"] += 1
                tag = f"[{counter['i']}/{total}] {u} | {mk}"
                try:
                    info = fut.result()
                    n_ok += 1
                    if info.get("skipped"):
                        _safe_print(f"{tag} SKIP: {info.get('reason')}"); continue
                    cost = info.get("est_cost_usd") or 0.0
                    total_cost += cost
                    if args.dry_run:
                        _safe_print(f"{tag} DRY  n={info.get('n_days') or info.get('n_events') or info.get('n_samples')}")
                    else:
                        n_units = info.get("n_days") or info.get("n_events") or info.get("n_samples")
                        cached = info.get("cached_units") or 0
                        _safe_print(
                            f"{tag} pt={info['prompt_tokens']} ct={info['completion_tokens']} "
                            f"cost=${cost:.4f} units={n_units} cached={cached}"
                        )
                except Exception as e:
                    n_err += 1
                    _safe_print(f"{tag} ERROR: {e}", file=sys.stderr)
                    if args.verbose:
                        traceback.print_exc()

    print("=" * 60)
    print(f"Done. ok={n_ok} err={n_err} cost=${total_cost:.4f}")
    if not args.dry_run:
        if task.startswith("proactive"):
            print("\nAggregating proactive summary...")
            aggregate_proactive.write_summary(
                judge_model_id=args.judge_model_id, run_judge=not args.no_judge,
            )
        else:
            print("\nAggregating emotion summary...")
            aggregate_emotion.write_summary()


def cmd_run(args):
    task = getattr(args, "task", "survey")
    if task in _TASK_RUNNER:
        _run_task_variant(args, task)
        return
    if task != "survey":
        print(f"Unknown task: {task}", file=sys.stderr); sys.exit(2)

    # ---- Survey path ----
    conds, models = load_config()
    users = args.users or ["z8a2", "q9rr"]
    available = set(loader.discover_users())
    missing = [u for u in users if u not in available]
    if missing:
        print(f"WARNING: missing trajectory for: {missing}. Available: {sorted(available)}")
        users = [u for u in users if u in available]
        if not users:
            print("No usable users. Aborting."); return

    condition_keys = args.conditions or list(conds.keys())
    model_keys = args.models or list(models.keys())
    bad_conds = [c for c in condition_keys if c not in conds]
    bad_models = [m for m in model_keys if m not in models]
    if bad_conds:
        print(f"ERROR: unknown conditions: {bad_conds}. Known: {list(conds.keys())}"); sys.exit(2)
    if bad_models:
        print(f"ERROR: unknown models: {bad_models}. Known: {list(models.keys())}"); sys.exit(2)

    persona_data = loader.load_persona()
    questions = loader.load_questions()
    print(f"Users:      {users}")
    print(f"Conditions: {condition_keys}")
    print(f"Models:     {model_keys}")
    print(f"Total runs: {len(users) * len(condition_keys) * len(model_keys)}")
    print(f"Dry run:    {args.dry_run}")
    print("=" * 60)

    ablations = list(getattr(args, "ablations", None) or [None])
    total = len(users) * len(condition_keys) * len(model_keys) * len(ablations)
    work = [(u, ck, mk, ab) for ab in ablations
                              for ck in condition_keys
                              for mk in model_keys
                              for u in users]
    workers = _parallel_workers()
    print(f"Ablations:  {ablations}")
    print(f"Parallel:   {workers} worker(s)  (total={total})")
    n_ok = n_err = 0; n_missing_letters = 0; total_cost = 0.0
    counter = {"i": 0}

    def _do(u: str, ck: str, mk: str, ab: str | None):
        return run_one(
            user_slug=u, condition_key=ck, model_key=mk,
            conds=conds, models=models, persona_data=persona_data,
            questions=questions, dry_run=args.dry_run,
            ablation_id=ab,
        )

    def _report(tag: str, info: dict):
        nonlocal n_ok, n_missing_letters, total_cost
        n_ok += 1
        if args.dry_run:
            _safe_print(f"{tag} DRY: user_chars={info['user_chars']} ~tokens={info['approx_user_tokens']}")
        else:
            cached_tag = " (cached)" if info.get("cached") else ""
            miss = info.get("n_missing", 0); n_missing_letters += miss
            cost = info.get("est_cost_usd") or 0.0; total_cost += cost
            _safe_print(
                f"{tag}{cached_tag} pt={info['prompt_tokens']} "
                f"ct={info['completion_tokens']} "
                f"cost=${cost:.4f} missing={miss}"
            )

    def _tag(i, u, ck, mk, ab):
        ab_part = f" | {ab}" if ab else ""
        return f"[{i}/{total}] {u} | {ck} | {mk}{ab_part}"

    if workers == 1:
        for u, ck, mk, ab in work:
            counter["i"] += 1
            tag = _tag(counter['i'], u, ck, mk, ab)
            try:
                _report(tag, _do(u, ck, mk, ab))
            except Exception as e:
                n_err += 1
                _safe_print(f"{tag} ERROR: {e}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_do, u, ck, mk, ab): (u, ck, mk, ab) for u, ck, mk, ab in work}
            for fut in as_completed(futs):
                u, ck, mk, ab = futs[fut]
                counter["i"] += 1
                tag = _tag(counter['i'], u, ck, mk, ab)
                try:
                    _report(tag, fut.result())
                except Exception as e:
                    n_err += 1
                    _safe_print(f"{tag} ERROR: {e}", file=sys.stderr)
                    if args.verbose:
                        traceback.print_exc()
    print("=" * 60); print(f"Done. ok={n_ok} err={n_err}")
    if not args.dry_run:
        print(f"Total cost (estimated): ${total_cost:.4f}")
        print(f"Total missing letters across all runs: {n_missing_letters}")
        print("\nAggregating summary CSVs...")
        aggregate.write_summary()


def build_parser():
    ap = argparse.ArgumentParser(prog="eval", description="JarvisBench eval CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-users", help="List users that have synthetic_trajectory.json")
    p.set_defaults(func=cmd_list_users)

    p = sub.add_parser("list-config", help="List configured conditions and models")
    p.set_defaults(func=cmd_list_config)

    p = sub.add_parser("eligibility", help="Print eligibility tables and write eligibility CSV/MD/TXT")
    p.set_defaults(func=cmd_eligibility)

    p_run = sub.add_parser("run", help="Run a task")
    p_run.add_argument(
        "--task",
        choices=["survey", "proactive_text", "proactive_image",
                 "emotion_text", "emotion_image"],
        default="survey",
    )
    p_run.add_argument("--users", nargs="+", default=None)
    p_run.add_argument("--conditions", nargs="+", default=None,
                       help="(survey only)")
    p_run.add_argument("--models", nargs="+", default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--judge-model-id", default="gpt-5-mini",
                       help="(proactive only) judge model id for reason matching")
    p_run.add_argument("--no-judge", action="store_true",
                       help="(proactive only) skip the LLM judge step")
    p_run.add_argument("--ablations", nargs="+", default=None,
                       help="(survey only) Domain Transfer ablation ids, e.g. "
                            "drop_food_dining keep_work_study. None = baseline.")
    p_run.set_defaults(func=cmd_run)

    p_pub = sub.add_parser("publish",
        help="Organize _summary/ flat files into _results/{task}/N{n_users}_M{n_models}/ tree")
    p_pub.set_defaults(func=cmd_publish)

    p_agg = sub.add_parser("aggregate", help="Recompute _summary CSVs from existing runs")
    p_agg.add_argument(
        "--task",
        choices=["survey", "proactive", "emotion", "all"],
        default="survey",
    )
    p_agg.add_argument("--judge-model-id", default="gpt-5-mini")
    p_agg.add_argument("--no-judge", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
