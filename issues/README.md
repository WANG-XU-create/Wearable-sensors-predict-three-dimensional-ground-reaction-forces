# Issues（本地）

本 repo 的 issue tracker 是**本地 markdown 文件**，不依赖 GitHub / `gh`。

- 每个 issue 一个文件：`issues/NNNN-<slug>.md`（`NNNN` 为 4 位递增编号）。
- 元数据（状态 / 标签 / assignee / 依赖）写在文件顶部 YAML frontmatter。
- 完整约定见 [`docs/agents/issue-tracker.md`](../docs/agents/issue-tracker.md)。
- 新建 issue 时复制 [`TEMPLATE.md`](./TEMPLATE.md)。
- 编号从 `0004` 起，延续原 GitHub issue 的 ticket 号（#1–#3 为历史已完成项，见 git 历史）。

## 快速操作

| 操作 | 命令 |
|---|---|
| 列出所有 issue | `ls issues/[0-9]*-*.md` |
| 读一个 issue | `Read issues/0001-*.md` |
| 按标签过滤 | `grep -l "needs-triage" issues/*.md` |
| 只看 open | `grep -l "^status: open" issues/*.md` |
