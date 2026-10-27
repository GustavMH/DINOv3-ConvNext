#!/usr/bin/env python3
"""fetch_weights.py — download DINOv3 weights from the Meta email and verify them.

Usage
-----
1. Paste the *whole* email body (the "name on one line, https://... on the next"
   block) into `a3/weights.urls`. Nothing else needs editing.
2. python3 scripts/fetch_weights.py                 # downloads to a3/weights/
   python3 scripts/fetch_weights.py --dest /data/w   # somewhere with space
   python3 scripts/fetch_weights.py --only sat493m   # filter by substring

Why this instead of a browser: the URLs are presigned bearer tokens that expire
(hours-to-days). wget is required by Meta's README for exactly this reason, and
`-c` lets you resume the 27 GB ViT-7B instead of restarting it.

Verification: Meta's filenames embed the first 8 hex digits of the SHA-256, e.g.
`dinov3_vitl16_pretrain_sat493m-<8hex>.pth`. Every download is checked against
that. `torch.hub.load(..., check_hash=True)` does the same check.

NOTE: the URL file is a secret. Do not commit it, do not paste logs with
`set -x` on, do not share the .urls file.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

PTH = re.compile(r"^(?P<name>[\w.+-]+\.pth)\s*$")
URL = re.compile(r"^(?P<url>https://\S+)\s*$")


def parse_email(text: str) -> list[tuple[str, str]]:
    """Pair each `foo.pth` line with the `https://` line that follows it."""
    pairs, pending = [], None
    for raw in text.splitlines():
        line = raw.strip()
        m = PTH.match(line)
        if m:
            pending = m.group("name")
            continue
        m = URL.match(line)
        if m and pending:
            pairs.append((pending, m.group("url")))
            pending = None
    return pairs


def expected_hash(name: str) -> str | None:
    m = re.search(r"-([0-9a-f]{8})\.pth$", name)
    return m.group(1) if m else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="weights.urls", type=Path)
    ap.add_argument("--dest", default="weights", type=Path)
    ap.add_argument("--only", default=None, help="substring filter on filename")
    ap.add_argument("--verify-only", action="store_true", help="skip downloads")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.urls.exists():
        print(f"!! {args.urls} not found — paste the email body there first", file=sys.stderr)
        return 2

    pairs = parse_email(args.urls.read_text())
    if not pairs:
        print("!! no (filename, url) pairs found in the url file", file=sys.stderr)
        return 2
    if args.only:
        pairs = [(n, u) for n, u in pairs if args.only in n]

    args.dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, url in pairs:
        dst = args.dest / name
        print(f"\n=== {name}")
        if args.dry_run:
            print("    (dry run)")
            continue
        if not args.verify_only and not (dst.exists() and dst.stat().st_size > 0):
            # -q keeps the presigned URL out of stdout; -c resumes a partial file.
            r = subprocess.run(
                ["wget", "-c", "-q", "--show-progress", "-O", str(dst), "--", url],
                stderr=subprocess.PIPE,
            )
            if r.returncode != 0:
                # print only the tail, with any Signature= token redacted
                msg = r.stderr.decode(errors="replace")[-400:]
                msg = re.sub(r"(Signature=)[^&\s]*", r"\1<redacted>", msg)
                print(f"!! wget failed ({r.returncode}): {msg}", file=sys.stderr)
                continue
        if not dst.exists():
            print(f"!! missing {dst}", file=sys.stderr)
            continue
        size = dst.stat().st_size
        total += size
        digest = sha256(dst)
        exp = expected_hash(name)
        ok = exp is not None and digest.startswith(exp)
        print(f"    {human(size):>10}  sha256 {digest[:16]}…  "
              f"{'OK (matches -%s)' % exp if ok else 'MISMATCH (expected %s)' % exp}")

    if total:
        print(f"\ntotal on disk: {human(total)}")
    print("\nnext: python3 scripts/inspect_weights.py weights/*.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
