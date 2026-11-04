"""Pretty-print aligned tables for human-readable summary reports.

Tries to use `tabulate` if installed; otherwise falls back to a hand-rolled
column-padding renderer so the pipeline never hard-fails.
"""
from __future__ import annotations

from typing import Any, Iterable

try:
    from tabulate import tabulate as _tabulate  # type: ignore
    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _rows_as_strings(rows: list[list[Any]]) -> list[list[str]]:
    return [[_fmt_cell(c) for c in r] for r in rows]


def _ascii_grid(headers: list[str], rows: list[list[Any]]) -> str:
    str_rows = _rows_as_strings(rows)
    cols = len(headers)
    widths = [len(headers[i]) for i in range(cols)]
    for r in str_rows:
        for i, c in enumerate(r):
            if i < cols and len(c) > widths[i]:
                widths[i] = len(c)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda row: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
    out = [sep, line(headers), sep]
    for r in str_rows:
        out.append(line(r))
    out.append(sep)
    return "\n".join(out)


def render_table(
    headers: list[str],
    rows: list[list[Any]],
    fmt: str = "txt",
) -> str:
    """fmt: 'txt' (ASCII grid) or 'md' (GitHub markdown)."""
    if fmt == "md":
        if _HAS_TABULATE:
            return _tabulate(_rows_as_strings(rows), headers=headers, tablefmt="github")
        # manual md
        head = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(_fmt_cell(c) for c in r) + " |" for r in rows]
        return "\n".join([head, sep] + body)
    # txt
    if _HAS_TABULATE:
        return _tabulate(_rows_as_strings(rows), headers=headers, tablefmt="grid")
    return _ascii_grid(headers, rows)


def df_to_table(df, fmt: str = "txt") -> str:
    headers = list(df.columns)
    rows = df.values.tolist()
    return render_table(headers, rows, fmt=fmt)


def write_report(
    path_md,
    path_txt,
    title: str,
    sections: list[tuple[str, list[str], list[list[Any]]]],
) -> None:
    """sections: list of (section_title, headers, rows)."""
    md_parts = [f"# {title}\n"]
    txt_parts = ["=" * 72, title, "=" * 72]
    for sec_title, headers, rows in sections:
        md_parts.append(f"\n## {sec_title}\n")
        md_parts.append(render_table(headers, rows, fmt="md"))
        md_parts.append("")
        txt_parts.append("")
        txt_parts.append(sec_title)
        txt_parts.append("-" * len(sec_title))
        txt_parts.append(render_table(headers, rows, fmt="txt"))
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_parts).rstrip() + "\n")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_parts).rstrip() + "\n")
