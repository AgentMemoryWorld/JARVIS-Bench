"""Domain Transfer task — load mapping + build event filters.

Setting S1 ("drop_<bucket>"): remove all events whose domain is in the bucket.
Setting S2 ("keep_<bucket>"): remove all events whose domain is in ANY OTHER
                              bucket. Long-tail events (not in any bucket)
                              are always kept regardless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "domain_mapping.yaml"


def load_mapping() -> dict[str, list[str]]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("buckets", {})


def list_ablation_ids() -> list[str]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(cfg.get("ablations_drop", []) + cfg.get("ablations_keep", []))


def parse_ablation(ablation_id: str) -> tuple[str, str]:
    """Return (mode, bucket). mode in {'drop','keep'}."""
    if ablation_id.startswith("drop_"):
        return "drop", ablation_id[len("drop_"):]
    if ablation_id.startswith("keep_"):
        return "keep", ablation_id[len("keep_"):]
    raise ValueError(f"Unknown ablation id: {ablation_id}")


def build_event_filter(ablation_id: str) -> Callable[[dict[str, Any]], bool]:
    """Return event_domain_filter(event)->bool for the given ablation.

    None ablation_id (or 'baseline') returns a filter that keeps everything.
    """
    if not ablation_id or ablation_id == "baseline":
        return lambda e: True

    mode, bucket = parse_ablation(ablation_id)
    mapping = load_mapping()
    if bucket not in mapping:
        raise ValueError(
            f"Ablation '{ablation_id}' references unknown bucket '{bucket}'. "
            f"Known buckets: {list(mapping)}"
        )

    bucket_doms = set(mapping[bucket])
    all_bucket_doms = {d for doms in mapping.values() for d in doms}

    if mode == "drop":
        # Remove events whose domain is in this bucket.
        def f(e: dict[str, Any]) -> bool:
            return e.get("domain") not in bucket_doms
        return f
    elif mode == "keep":
        # Keep events whose domain is in this bucket OR is long-tail (not in any bucket).
        def f(e: dict[str, Any]) -> bool:
            d = e.get("domain")
            return d in bucket_doms or d not in all_bucket_doms
        return f
    raise ValueError(f"unknown mode: {mode}")
