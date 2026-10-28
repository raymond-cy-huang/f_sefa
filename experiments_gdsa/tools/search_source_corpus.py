#!/usr/bin/env python3
"""Search the extracted corpus and report source paths with PDF page context."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--context", type=int, default=220)
    args = parser.parse_args()

    payload = json.loads(args.index.read_text(encoding="utf-8"))
    pattern = re.compile(args.query, flags=re.IGNORECASE)
    found = 0
    for source in payload["sources"]:
        output = source.get("output_path")
        if source.get("status") != "ok" or not output:
            continue
        text = Path(output).read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            page = text.count("\f", 0, match.start()) + 1 if source.get("pages") else None
            start = max(0, match.start() - args.context)
            end = min(len(text), match.end() + args.context)
            print(
                json.dumps(
                    {
                        "source_path": source["source_path"],
                        "page": page,
                        "context": compact(text[start:end]),
                    },
                    ensure_ascii=False,
                )
            )
            found += 1
            if found >= args.max_results:
                return


if __name__ == "__main__":
    main()
