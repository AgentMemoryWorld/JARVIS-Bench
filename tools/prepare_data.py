"""Lay out the JARVIS-Bench data bundle the way eval/ expects it.

The data is hosted on Harvard Dataverse (https://doi.org/10.7910/DVN/8UKMK1) as
jarvis-bench_full.zip, which holds two archives:

  User.zip   User/{slug}/synthetic_trajectory.json, profile_persona.json, skeleton.json, ...
  Image.zip  Image/{slug}/Day{N}/<photo>

Run from the repository root:

  python tools/prepare_data.py /path/to/jarvis-bench_full.zip
  python tools/prepare_data.py /path/to/folder        # holds User.zip/Image.zip or User/ and Image/
  python tools/prepare_data.py ... --no-images        # Implicit Preference only (no photo tasks)

It writes

  synthetic/{slug}/synthetic_trajectory.json, profile_persona.json, skeleton.json
  Data/processed/image/{slug}/Day{N}/<photo>
  Data/processed/survey.json   (a copy of ground_truth/survey_ground_truth.json)

and prints a short summary. Existing files are overwritten. Only the standard
library is used.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "synthetic"
IMAGES = ROOT / "Data" / "processed" / "image"
SURVEY_SRC = ROOT / "ground_truth" / "survey_ground_truth.json"
SURVEY_DST = ROOT / "Data" / "processed" / "survey.json"
USER_FILES = ("synthetic_trajectory.json", "profile_persona.json", "skeleton.json")


def _safe_parts(name: str) -> tuple[str, ...] | None:
    """Path parts of an archive member, or None for junk / unsafe entries."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "__MACOSX" in p.parts or p.name.startswith("._") or p.name == ".DS_Store":
        return None
    return p.parts


def _open_inner(source: Path, stem: str, stack: list) -> zipfile.ZipFile | Path | None:
    """Return the User/Image data as an open ZipFile, an extracted folder, or None."""
    if source.is_dir():
        if (source / f"{stem}.zip").is_file():
            zf = zipfile.ZipFile(source / f"{stem}.zip"); stack.append(zf); return zf
        if (source / stem).is_dir():
            return source / stem
        return None
    outer = zipfile.ZipFile(source); stack.append(outer)
    member = next((m for m in outer.infolist()
                   if _safe_parts(m.filename) and PurePosixPath(m.filename).name == f"{stem}.zip"), None)
    if member is None:
        return None
    # Copy the inner archive to a temporary file first: reading members of a
    # compressed archive nested inside another one is far too slow otherwise.
    tmp = tempfile.TemporaryFile(); stack.append(tmp)
    with outer.open(member) as f:
        shutil.copyfileobj(f, tmp, 1 << 20)
    tmp.seek(0)
    zf = zipfile.ZipFile(tmp); stack.append(zf)
    return zf


def _members(src: zipfile.ZipFile | Path, top: str):
    """Yield (parts below `top`, reader) for every regular file."""
    if isinstance(src, Path):
        for f in sorted(src.rglob("*")):
            if f.is_file():
                parts = _safe_parts(f.relative_to(src).as_posix())
                if parts:
                    yield parts, f.read_bytes
        return
    for m in src.infolist():
        if m.is_dir():
            continue
        parts = _safe_parts(m.filename)
        if not parts:
            continue
        if parts[0] == top:
            parts = parts[1:]
        if parts:
            yield parts, (lambda m=m: src.read(m))


def prepare_users(src) -> list[str]:
    files: dict[str, dict[str, object]] = {}
    for parts, read in _members(src, "User"):
        if len(parts) == 2 and parts[1] in USER_FILES:
            files.setdefault(parts[0], {})[parts[1]] = read
    users = sorted(slug for slug, fs in files.items() if "synthetic_trajectory.json" in fs)
    for slug in users:
        out = SYNTHETIC / slug
        out.mkdir(parents=True, exist_ok=True)
        for name, read in files[slug].items():
            (out / name).write_bytes(read())
    return users


def prepare_images(src) -> tuple[int, dict[str, int]]:
    n, kinds = 0, {}
    for parts, read in _members(src, "Image"):
        if len(parts) < 3:
            continue
        out = IMAGES.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(read())
        n += 1
        ext = out.suffix.lower().lstrip(".")
        kinds[ext] = kinds.get(ext, 0) + 1
    return n, kinds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="jarvis-bench_full.zip, or a folder holding User/Image (zip or folders)")
    ap.add_argument("--no-images", action="store_true", help="skip the photos (enough for Implicit Preference Reasoning)")
    a = ap.parse_args()
    if not a.source.exists():
        sys.exit(f"not found: {a.source}")
    if not SURVEY_SRC.is_file():
        sys.exit(f"missing {SURVEY_SRC.relative_to(ROOT)}; run from a full clone of the repository")

    stack: list = []
    try:
        users_src = _open_inner(a.source, "User", stack)
        if users_src is None:
            sys.exit("could not find User.zip or User/ in the source")
        users = prepare_users(users_src)
        print(f"synthetic/: {len(users)} users with a trajectory")

        if not a.no_images:
            image_src = _open_inner(a.source, "Image", stack)
            if image_src is None:
                sys.exit("could not find Image.zip or Image/ in the source (use --no-images to skip)")
            n, kinds = prepare_images(image_src)
            print(f"Data/processed/image/: {n} photos {dict(sorted(kinds.items()))}")
    finally:
        for obj in reversed(stack):
            obj.close()

    SURVEY_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SURVEY_SRC, SURVEY_DST)
    answers = json.load(open(SURVEY_DST, encoding="utf-8"))
    missing = [u for u in users if u not in answers]
    print(f"Data/processed/survey.json: answers for {len(answers)} users"
          + (f"; no answers for {missing}" if missing else ""))


if __name__ == "__main__":
    main()
