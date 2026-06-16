#!/usr/bin/env python3
"""扫描 papers/ 下所有笔记的 YAML frontmatter，生成 INDEX.md（按日期）与 TAGS.md（按标签）。

用法:
    python scripts/build_index.py

仅依赖标准库，不需要安装第三方包。
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_scalar(value: str):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [v.strip().strip('"').strip("'") for v in inner.split(",")]
    return value.strip().strip('"').strip("'")


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    data: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        data[key.strip()] = parse_scalar(raw)
    return data


def collect() -> list[dict]:
    papers = []
    for md in sorted(PAPERS_DIR.rglob("*.md")):
        meta = parse_frontmatter(md.read_text(encoding="utf-8"))
        if not meta.get("title"):
            continue
        meta["_path"] = md.relative_to(ROOT).as_posix()
        if isinstance(meta.get("tags"), str):
            meta["tags"] = [meta["tags"]] if meta["tags"] else []
        papers.append(meta)
    return papers


def fmt_link(p: dict) -> str:
    title = p.get("title", "Untitled")
    path = p["_path"]
    arxiv = p.get("arxiv", "")
    arxiv_part = f" · [arXiv:{arxiv}]({p.get('url', '')})" if arxiv else ""
    return f"[{title}]({path}){arxiv_part}"


def build_index(papers: list[dict]) -> str:
    by_date = sorted(papers, key=lambda p: str(p.get("date_read", "")), reverse=True)
    lines = ["# 📅 论文索引（按阅读日期）", "", f"共 {len(papers)} 篇。由 `scripts/build_index.py` 自动生成，请勿手动编辑。", ""]
    current = None
    for p in by_date:
        date = str(p.get("date_read", "未知日期"))
        month = date[:7] if len(date) >= 7 else date
        if month != current:
            current = month
            lines.append(f"\n## {month}\n")
        tags = ", ".join(f"`{t}`" for t in p.get("tags", []) if t)
        rating = "⭐" * int(p["rating"]) if str(p.get("rating", "")).isdigit() else ""
        lines.append(f"- **{date}** — {fmt_link(p)} {rating}")
        if tags:
            lines.append(f"  - {tags}")
    return "\n".join(lines) + "\n"


def build_tags(papers: list[dict]) -> str:
    tag_map: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        for t in p.get("tags", []):
            if t:
                tag_map[t].append(p)
    lines = ["# 🏷️ 标签索引", "", "由 `scripts/build_index.py` 自动生成，请勿手动编辑。", ""]
    for tag in sorted(tag_map, key=lambda t: (-len(tag_map[t]), t)):
        lines.append(f"\n## `{tag}` ({len(tag_map[tag])})\n")
        for p in sorted(tag_map[tag], key=lambda p: str(p.get("date_read", "")), reverse=True):
            lines.append(f"- {p.get('date_read', '')} — {fmt_link(p)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    papers = collect()
    (ROOT / "INDEX.md").write_text(build_index(papers), encoding="utf-8")
    (ROOT / "TAGS.md").write_text(build_tags(papers), encoding="utf-8")
    print(f"Indexed {len(papers)} papers -> INDEX.md, TAGS.md")


if __name__ == "__main__":
    main()
