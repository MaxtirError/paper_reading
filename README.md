# 📚 Paper Reading

个人论文阅读笔记仓库。核心是**用 GitHub Copilot agent 精读 paper**：把论文读进上下文，然后**回答我的问题**；一轮问答结束后，再把 QA 沉淀成笔记归档、检索（主要是 CV 方向）。

> 重点是「读 + 问答」，记笔记只是副产物。提问的是我，不是 agent。

## 快速开始

在本仓库工作区打开 Copilot Chat：

**1. 让 agent 读论文（强制下载 PDF + 全文扫描）**
```
/read-paper https://arxiv.org/abs/2503.11651
```
agent 会运行 `scripts/fetch_paper.py` 把 PDF 下到 `cache/`、抽取全文并整篇读入，然后回一句确认就停下，等我提问。
> ⛔ 不退化到摘要。拿不到 PDF 时 agent 会让我手动把 PDF 拖入聊天。

**2. 我开始提问，agent 基于 PDF 全文作答**
```
它和 DUSt3R 的核心区别是什么？
global attention 那块显存怎么 scale？
```

**3. 确认没有其他问题后让它归档**
```
没有其他问题了，帮我把这次问答记成笔记
```
agent 把真实 QA 写进 `papers/年/月/` 的笔记并刷新索引。笔记会上传 GitHub；`cache/` 里的 PDF 不会。

## 仓库结构

```
paper_reading/
├── README.md                     # 你正在看的文件
├── WORKFLOW.md                   # 完整工作流方案（中文）
├── INDEX.md                      # 自动生成：按日期排列的全部论文
├── TAGS.md                       # 自动生成：按 tag 分组的论文
├── templates/
│   └── paper-note.md             # 单篇笔记模板
├── papers/
│   └── 2026/06/                  # 按 年/月 归档
│       └── 2026-06-16-xxx.md     # 文件名带日期前缀
├── cache/                        # 下载的 PDF + 抽取文本（被 .gitignore 忽略，不上传）
├── scripts/
│   ├── fetch_paper.py            # 下载 arXiv PDF 到 cache/ 并抽取全文
│   └── build_index.py            # 扫描 frontmatter 生成 INDEX/TAGS
├── .gitignore                    # 忽略 cache/ 等
└── .github/
    ├── copilot-instructions.md   # 仓库级 agent 行为约定
    └── prompts/
        └── read-paper.prompt.md  # /read-paper 可复用提示词
```

## 检索方式

- **按日期**：看 [INDEX.md](INDEX.md)，或直接浏览 `papers/年/月/` 目录。
- **按标签**：看 [TAGS.md](TAGS.md)（如 `3d-reconstruction`、`diffusion`、`detection`…）。
- **全文搜索**：VS Code 里 `Ctrl/Cmd+Shift+F` 搜关键词。

## 刷新索引

新增/修改笔记后运行：

```bash
python scripts/build_index.py
```
