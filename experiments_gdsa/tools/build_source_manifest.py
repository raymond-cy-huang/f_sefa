#!/usr/bin/env python3
"""Build a deterministic manifest for the authoritative GDSA source corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        stat = path.stat()
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    records = collect(root)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "file_count": len(records),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in records),
        "files": records,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["relative_path"])
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps({key: payload[key] for key in ("root", "file_count", "total_size_bytes")}, indent=2))


if __name__ == "__main__":
    main()
