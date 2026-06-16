# Paper Reading 工作流方案

> 一个用 GitHub Copilot agent **陪我精读论文**的方案：把 paper 读进上下文 → **我提问、agent 基于论文作答** → 问答结束后把这轮 QA 沉淀成笔记。本文是中文说明，骨架已在本仓库搭好。

> **重点是「读 + 问答」，不是自动记笔记。提问的是我（用户），不是 agent；笔记是问答的副产物。**

---

## 一、整体流程（交互式）

```
阶段 1　我用 /read-paper 给 arXiv 链接/ID
        │
        ▼
      agent 必须跑 scripts/fetch_paper.py 把 PDF 下到 cache/、抽取全文并整篇读入
      （ⴛ 不退化到摘要；拿不到 PDF 就让我手动拖入聊天）
        │
        ▼
      读完只回一句确认（标题 + 一句话主旨），然后停下来等我提问
        │
        ▼
阶段 2　【主循环】我逐个提问 ↔ agent 基于 PDF 全文作答（给出章节/公式/图表出处）
        │
        ▼  （我说“没有其他问题 / 结束 / 归档”时才进入下一步）
阶段 3　agent 把本次真实 QA 写进 papers/年/月/YYYY-MM-DD-slug.md
        │
        ▼
      运行 build_index.py → 刷新 INDEX.md（按日期）与 TAGS.md（按标签）
      （笔记上传 GitHub；cache/ 里的 PDF 不上传）
```

日常用法：
```
# 1) 让它下载并读整篇 PDF
/read-paper https://arxiv.org/abs/2503.11651
# 2) 我提问（任意多轮）
它和 DUSt3R 的核心区别是什么？
# 3) 确认无问题后归档
没有其他问题了，帮我把这次问答记成笔记
```

---

## 二、问题 1：PDF 如何传递给 agent（强制下载 + 全文扫描）

**硬性流程：必须把 PDF 下载到本地 cache/ 并整篇扫描，不接受退化到摘要。**

1. **下载**：agent 运行 `python scripts/fetch_paper.py <arxiv链接或ID>`：
   - 从 arXiv 下载 PDF 到 `cache/<id>.pdf`；
   - 抽取全文到 `cache/<id>.txt`（依次尝试 PyMuPDF / pdfminer.six / pdftotext）。
2. **全文读入**：agent 用 `read_file` 读取 `cache/<id>.txt` 的**完整内容**（大文件分段读，覆盖全文）。
3. ⴛ **严禁退化**：不接受只看 arXiv 摘要/abstract 页来回答。
4. **拿不到就求助**：脚本失败（下载失败 / 非 PDF / 文本过短）时，agent **不用摘要凑合**，而是让我手动下载并把 PDF 拖入聊天（或放到 `cache/`）。

> `cache/` 已被 `.gitignore` 忽略：PDF / 抽取文本 **不上传 GitHub**，只有笔记 Markdown 会上传。
>
> 本地也可手动跑：`python scripts/fetch_paper.py 2503.11651`。需装一个抽取器：`pip install pymupdf`（推荐）。

---

## 三、问题 2：归档结构（兼顾日期 + 标签两种检索）

### 目录结构
```
papers/
└── 2026/            # 年
    └── 06/          # 月（两位）
        └── 2026-06-16-vggt.md   # 文件名带完整日期前缀 + slug
```
- **按日期查**：目录天然有序；`INDEX.md` 按阅读日期倒序列出全部。
- **按标签查**：每篇 frontmatter 里有 `tags`，`TAGS.md` 自动按标签分组。
- **全文搜**：VS Code `Ctrl/Cmd+Shift+F`。

### 每篇笔记的元数据（YAML frontmatter）
见 [templates/paper-note.md](templates/paper-note.md)，关键字段：

| 字段 | 作用 |
|------|------|
| `title` / `authors` / `year` / `venue` | 基本信息 |
| `arxiv` / `url` / `code` | 溯源链接 |
| `date_read` | 按日期检索的依据 |
| `category` | 单一一级方向（如 `3d-reconstruction`） |
| `tags` | 3–6 个细粒度标签（按标签检索的依据） |
| `rating` | 1–5 主观重要度，便于回头筛重点 |
| `status` | `read` / `skimmed` / `to-read` |

### 正文以【真实 QA】为主体
- **我的问答记录（QA）**：按对话顺序保留我的原始问法 + agent 的解答（这是笔记的核心）。
- **我的收获 / 结论**：用我自己的话写下判断与可借鉴点。
- **关联工作 / 关键摘抄**（可选）。

为什么以 QA 为主：记录的是「我当时真正在乎的点」，回头看比一份八股总结更有价值。

### 标签规范（避免碎片化）
- 全小写、连字符：`open-vocabulary`、`3d-reconstruction`。
- 新建前先看 `TAGS.md` 复用已有标签，别造同义词（如 `3d` vs `3d-reconstruction`）。

---

## 四、索引自动生成

[scripts/build_index.py](scripts/build_index.py) 只用标准库，扫描所有笔记的 frontmatter，生成：
- `INDEX.md`：按月份分组、按日期倒序、带星级。
- `TAGS.md`：按标签分组，热门标签在前。

新增/修改笔记后运行一次：
```bash
python scripts/build_index.py
```
（agent 在归档时也会自动跑这一步。）

---

## 五、本仓库已生成的文件

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 仓库入口与用法 |
| [templates/paper-note.md](templates/paper-note.md) | 单篇笔记模板 |
| [papers/2026/06/2026-06-16-vggt.md](papers/2026/06/2026-06-16-vggt.md) | **示例笔记**（演示格式，可删） |
| [scripts/build_index.py](scripts/build_index.py) | 索引生成脚本 |
| [INDEX.md](INDEX.md) / [TAGS.md](TAGS.md) | 自动生成的索引 |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | 仓库级 agent 行为约定 |
| [.github/prompts/read-paper.prompt.md](.github/prompts/read-paper.prompt.md) | `/read-paper` 可复用提示词 |

---

## 六、推送到 GitHub

```bash
cd paper_reading
git init
git add .
git commit -m "chore: bootstrap paper-reading workflow"
# 在 GitHub 新建空仓库后：
git remote add origin git@github.com:<你>/paper_reading.git
git push -u origin main
```

---

## 七、可选增强（按需再加，先不做）
- GitHub Actions：push 时自动跑 `build_index.py` 并提交，免手动。
- 每篇加 `bibtex` 字段，方便写论文时引用。
- 加 `to-read` 队列文件，管理待读清单。
