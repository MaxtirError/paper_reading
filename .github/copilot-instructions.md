# Copilot 仓库指引：Paper Reading

这是一个个人论文阅读笔记仓库，方向以计算机视觉（CV）为主。

## 核心理念：读 + 答，不是自动记笔记
- 这个仓库是用来**精读论文**的，重点是把论文读进上下文，然后**回答用户提出的问题**。
- **提问的是用户，不是 agent。** 不要主动抓取“核心问题/关键词”、不要替用户编问题、不要在读完后就甩一份长总结。
- 读完论文后只做一句简短确认（标题 + 一句话主旨 + 邀请用户提问），然后进入问答主循环。
- 回答要基于论文、给出处（章节/公式/图表/原文短句）；论文没说的标注为推断，不臆造。
- **笔记是问答的副产物**：仅当用户明确说“结束/归档/记笔记”时，才把这轮真实 QA 整理成笔记并刷新索引。

## 读 paper：强制下载 PDF 全文扫描（硬性流程）
- 用户用 `/read-paper <arxiv链接或ID>` 触发。**必须**运行 `python scripts/fetch_paper.py <arxiv链接或ID>` 把 PDF 下到 `cache/<id>.pdf` 并抽取全文到 `cache/<id>.txt`。
- 然后用 `read_file` 读取 `cache/<id>.txt` 的**完整内容**（大文件分段读，覆盖全文）。
- ⛔ **不接受退化到摘要/abstract**。必须基于 PDF 全文回答。
- 拿不到 PDF（脚本失败）时，**不要用摘要凑合**：告诉用户手动下载并把 PDF 拖入聊天，或放到 `cache/`，然后等待。
- `cache/` 已被 `.gitignore` 忽略：PDF/文本**不上传 GitHub**，只有笔记 Markdown 会上传。

## 归档规则
- 每篇论文一个 Markdown 文件，路径为 `papers/<年>/<月两位>/<YYYY-MM-DD>-<slug>.md`。
- 文件必须以 YAML frontmatter 开头，字段见 `templates/paper-note.md`。
- 正文主体是「我的问答记录（QA）」：按对话顺序保留用户的原始问题 + 解答；其后是「我的收获/结论」「关联工作」「关键摘抄」。

## 标签规范
- `tags` 全部小写、用连字符（如 `3d-reconstruction`、`open-vocabulary`）。
- 新建笔记前先查看 `TAGS.md`，尽量复用已有标签，避免同义词碎片化。
- `category` 是单一一级方向；`tags` 是 3-6 个细粒度标签。

## 索引
- 新增或修改笔记后，运行 `python scripts/build_index.py` 重新生成 `INDEX.md` 和 `TAGS.md`。
- 不要手动编辑 `INDEX.md` / `TAGS.md`（它们是自动生成的）。
