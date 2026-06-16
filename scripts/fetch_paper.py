#!/usr/bin/env python3
"""下载 arXiv 论文 PDF 到本地 cache 并抽取全文文本，供 agent 全文扫描。

用法:
    python scripts/fetch_paper.py <arxiv链接或ID>

行为:
- 解析出 arXiv id（接受 abs/pdf/html 链接或裸 id）。
- 下载 PDF 到 cache/<id>.pdf（cache 已被 .gitignore 忽略，不会上传）。
- 抽取纯文本到 cache/<id>.txt（agent 读取此文件来回答问题）。
- **不接受退化到摘要**：拿不到 PDF 或抽不出文本时以非零退出码失败，
  并打印提示让用户手动上传 PDF。

文本抽取按可用性依次尝试：PyMuPDF(fitz) -> pdfminer.six -> pdftotext(poppler)。
"""
from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def parse_arxiv_id(arg: str) -> str:
    m = ARXIV_ID_RE.search(arg)
    if not m:
        sys.exit(f"[ERROR] 无法从输入解析 arXiv id: {arg!r}")
    return m.group(1) + (m.group(2) or "")


def download_pdf(arxiv_id: str, dest: Path) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 paper-reading-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"[ERROR] 下载 PDF 失败 ({url}): {e}\n"
            "请手动下载该 PDF 并拖入聊天，或放到 cache/ 目录后重试。"
        )
    if not data[:5] == b"%PDF-":
        sys.exit(
            f"[ERROR] 下载内容不是有效 PDF ({url})。\n"
            "请手动下载该 PDF 并拖入聊天。"
        )
    dest.write_bytes(data)


def extract_text(pdf: Path, txt: Path) -> int:
    # 1) PyMuPDF
    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf)
        text = "\n".join(page.get_text() for page in doc)
        txt.write_text(text, encoding="utf-8")
        return len(text)
    except ImportError:
        pass

    # 2) pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore

        text = pdfminer_extract(str(pdf))
        txt.write_text(text, encoding="utf-8")
        return len(text)
    except ImportError:
        pass

    # 3) pdftotext (poppler)
    try:
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=True)
        return len(txt.read_text(encoding="utf-8", errors="ignore"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    sys.exit(
        "[ERROR] 没有可用的 PDF 文本抽取器。请安装其一：\n"
        "  pip install pymupdf      # 推荐\n"
        "  pip install pdfminer.six\n"
        "  或安装 poppler 的 pdftotext\n"
        "PDF 已下载到 cache/，可手动抽取或直接把 PDF 拖入聊天。"
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("用法: python scripts/fetch_paper.py <arxiv链接或ID>")
    arxiv_id = parse_arxiv_id(sys.argv[1])
    CACHE.mkdir(exist_ok=True)
    pdf = CACHE / f"{arxiv_id}.pdf"
    txt = CACHE / f"{arxiv_id}.txt"

    if not pdf.exists():
        download_pdf(arxiv_id, pdf)
    n = extract_text(pdf, txt)
    if n < 500:
        sys.exit(
            f"[ERROR] 抽取到的文本过短 ({n} 字符)，可能是扫描版/图片型 PDF。\n"
            "请手动确认或把 PDF 拖入聊天由 agent 直接读取。"
        )
    print(f"[OK] arxiv_id={arxiv_id}")
    print(f"[OK] pdf  -> {pdf.relative_to(ROOT)}")
    print(f"[OK] text -> {txt.relative_to(ROOT)} ({n} 字符)")


if __name__ == "__main__":
    main()
