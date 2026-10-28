#!/usr/bin/env python3
"""Extract a searchable cache from the complete authoritative source corpus."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


SUPPORTED = {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}


def pdf_text(path: Path) -> tuple[str, int | None]:
    proc = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        check=True,
        capture_output=True,
    )
    info = subprocess.run(["pdfinfo", str(path)], check=False, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", info.stdout, flags=re.MULTILINE)
    return proc.stdout.decode("utf-8", errors="replace"), int(match.group(1)) if match else None


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        data = archive.read("word/document.xml")
    root = ElementTree.fromstring(data)
    paragraphs: list[str] = []
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for paragraph in root.iter(namespace + "p"):
        pieces = [node.text or "" for node in paragraph.iter(namespace + "t")]
        if pieces:
            paragraphs.append("".join(pieces))
    return "\n".join(paragraphs)


def html_text(path: Path) -> str:
    source = path.read_text(encoding="utf-8", errors="replace")
    source = re.sub(r"<(script|style)\b.*?</\1>", " ", source, flags=re.I | re.S)
    source = re.sub(r"<[^>]+>", " ", source)
    source = html.unescape(source)
    return re.sub(r"[ \t]+", " ", source)


def extract(path: Path) -> tuple[str, int | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # Some literature repositories contain publisher HTML saved with a
        # misleading .pdf suffix. Detect content instead of trusting the name.
        header = path.read_bytes()[:1024].lstrip().lower()
        if header.startswith((b"<html", b"<!doctype html")):
            return html_text(path), None
        return pdf_text(path)
    if suffix == ".docx":
        return docx_text(path), None
    if suffix in {".html", ".htm"}:
        return html_text(path), None
    return path.read_text(encoding="utf-8", errors="replace"), None


def safe_name(prefix: str, relative: Path) -> Path:
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_") or "unnamed" for part in relative.parts]
    return Path(prefix, *parts).with_suffix(Path(parts[-1]).suffix + ".txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--extra", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    sources: list[tuple[str, Path, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and ".git" not in path.relative_to(root).parts:
            sources.append(("literature_review_papers", path, path.relative_to(root)))
    for extra in args.extra:
        path = extra.resolve()
        sources.append(("f_sefa_docs", path, Path(path.name)))

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for corpus, source, relative in sources:
        row: dict[str, object] = {
            "corpus": corpus,
            "source_path": str(source),
            "relative_path": relative.as_posix(),
            "extension": source.suffix.lower(),
        }
        if source.suffix.lower() not in SUPPORTED:
            row.update(status="unsupported", output_path=None, characters=0, pages=None)
            rows.append(row)
            continue
        target = args.output / safe_name(corpus, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            text, pages = extract(source)
            target.write_text(text, encoding="utf-8")
            row.update(status="ok", output_path=str(target.resolve()), characters=len(text), pages=pages)
        except Exception as error:  # preserve every failure in the index
            row.update(status="error", output_path=None, characters=0, pages=None, error=repr(error))
        rows.append(row)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(rows),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "unsupported_count": sum(row["status"] == "unsupported" for row in rows),
        "total_characters": sum(int(row["characters"]) for row in rows),
        "sources": rows,
    }
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("source_count", "ok_count", "error_count", "unsupported_count", "total_characters")}, indent=2))


if __name__ == "__main__":
    main()
