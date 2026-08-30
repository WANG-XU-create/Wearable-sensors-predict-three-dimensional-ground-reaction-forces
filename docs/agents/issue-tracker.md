# Issue tracker: 本地 markdown 文件

这个 repo 的 issues 和 specs 存放在本地 `issues/` 目录。**不依赖 GitHub，也不使用 `gh`**——所有操作都是普通的文件读写。

## 存储格式

- 每个 issue 一个文件：`issues/NNNN-<slug>.md`，`NNNN` 为 4 位递增编号（新 issue 取现有最大编号 + 1）。
- 元数据放在文件顶部 YAML frontmatter：

```yaml
---
id: 0001
title: "..."           # 必需
status: open           # open | closed
labels: []             # 见 docs/agents/triage-labels.md
assignee: ""           # 空 = 未分配；claim 时填名字
blocked_by: []         # 阻塞本 issue 的编号列表，如 [0002, 0003]
part_of: ""            # wayfinder child 时填 map 的编号
created: 2026-08-30
updated: 2026-08-30
---
```

- body 是描述（背景 / 目标 / 验收标准）。
- 评论追加在 `## Comments` 小节内。

## Conventions

- **Create an issue**: 复制 `issues/TEMPLATE.md` 为 `issues/NNNN-<slug>.md`，填好 frontmatter 和 `## Description`。
- **Read an issue**: `Read issues/NNNN-<slug>.md`。frontmatter 即状态与 labels，body + `## Comments` 即全部内容，无需额外查询。
- **List issues**: `ls issues/[0-9]*-*.md`。
  - 按标签过滤：`grep -l "needs-triage" issues/*.md`
  - 只看 open：`grep -l "^status: open" issues/*.md`
  - 叠加过滤：`grep -l "^status: open" $(grep -l "needs-triage" issues/*.md)`
- **Comment on an issue**: 在目标文件 `## Comments` 小节末尾追加一节，并更新 `updated`：
  ```
  ### @作者 — 2026-08-30
  内容
  ```
- **Apply / remove labels**: 编辑 frontmatter 的 `labels` 列表，更新 `updated`。
- **Close**: 把 `status` 改为 `closed`（需要时先追加一条 closing comment），更新 `updated`。

不再从 `git remote` 推断任何东西，也不再需要任何远程凭据。

## Pull requests as a triage surface

**PRs as a request surface: no.** 本 repo 的 issue tracker 是本地文件，不处理 GitHub PRs。如将来需要，把 PR 相关信息手工记录为普通 issue 即可。

## When a skill says "publish to the issue tracker"

创建一个本地 issue 文件（见上方 **Create an issue**）。

## When a skill says "fetch the relevant ticket"

`Read issues/NNNN-<slug>.md`。

## Wayfinding operations

供 `/wayfinder` 使用。**map** 是单个 issue，以 **child** issues 作为 tickets。

- **Map**: 单个带 `wayfinder:map` label 的 issue，body 保存 Notes / Decisions-so-far / Fog 三节。
- **Child ticket**: 一个普通 issue，frontmatter `part_of` 填 map 的编号（如 `"0000"`），并在 body 顶部写 `Part of #0000`。Labels 加 `wayfinder:<type>`（`research`/`prototype`/`grilling`/`task`）。一旦被 claim，`assignee` 填 driving dev。
- **Blocking**: frontmatter 的 `blocked_by` 列表是 canonical 表达——只填**仍然 open** 的 blocker 编号（即 live gate）。当所有 blockers 都关闭时，ticket 即为 unblocked；把编号从列表里移除即解除阻塞。
- **Frontier query**: 找出 map 的 open children（`grep -l "part_of: \"0000\"" issues/*.md` 且 `status: open`），丢弃带 open blocker 的（`blocked_by` 非空）或带 assignee 的；按编号排序，第一个胜出。
- **Claim**: 把 `assignee` 填上当前 driving dev 的名字，更新 `updated`——session 的第一次写入。
- **Resolve**: 追加一条 answer comment，把 `status` 改为 `closed`，再向 map 的 Decisions-so-far 追加 context pointer（文件路径 + 简短结论）。
