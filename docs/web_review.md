# 网页与教材正文审阅

Parquet 是列式存储格式，不适合在文本编辑器中直接阅读。当前实验提供一个本地、只读、无外部依赖的 HTML 审阅导出。

## 浏览方式

- 审阅页：[打开本机审阅服务](http://127.0.0.1:8767/)。此入口需要本机服务运行；Remote SSH 场景如在外部浏览器打开，需要转发端口 **8767**。
- 可按 OpenStax / FineMath-4+ / OpenWebMath / FineWeb 来源筛选，搜索完整正文 / 教材标题 / 章节 / URL / ID、翻页、展开完整溯源信息、下载单篇 TXT。
- [直接看 OpenStax](http://127.0.0.1:8767/?source=openstax)：117 个实际评分的完整教材小节；[直接看 FineMath-4+](http://127.0.0.1:8767/?source=finemath)：1,000 篇实际评分的网页。原 1,000 篇 OpenWebMath 和 9,000 篇 FineWeb 保留。
- 同时展示 reference / SFT NLL、loss 下降量和相对下降比例，可分别排序。默认固定 seed 随机顺序，不按分数挑样本。
- 所有正文完整保留，只有左侧列表摘要截短。正文始终作为纯文本显示，不执行原网页 HTML，也不渲染 LaTeX。
- 单文件 HTML 可以下载后离线打开，无需服务、网络、模型或 Python。默认首次打开载入全部 11,117 篇正文；共用背景只保留一次。页面合并展示并不改变各实验候选池或来源比例。

## 产物与复现

### 服务与连接排查

- 2026-10-26：原 8766 服务仍运行且服务器本机 HTTP 200，但远程浏览器连小清单都超时、请求未到达服务；改用 8767 后浏览器可访问。旧 8766 链接不再作为入口。
- 当前使用用户级 systemd 服务 `data-picking-web-review.service`，仅绑定 `127.0.0.1:8767`，启用失败自动重启。不需要保留启动终端；仅提供审阅导出目录，不暴露项目根目录。
- 服务管理配置保存在 `~/.config/systemd/user/data-picking-web-review.service`。可用 `systemctl --user status data-picking-web-review.service --no-pager` 检查，或 `systemctl --user restart data-picking-web-review.service` 重启。
- SSH 重连后如打不开，先在 VS Code“端口”面板移除旧的失效转发，再转发远程端口 8767，并使用面板给出的实际本地地址（本地端口可能因冲突而不同）。可先打开小文件 `review_manifest.json` 检查连接，不必等待约 56 MiB 的全文页面。
- 后台服务与 SSH 转发是两个独立环节：服务正常不代表本地转发还有效。用户级服务也不能保证在用户完全登出或主机重启后、用户管理器尚未启动时可用。
- 也可下载完整 `index.html` 后在本地浏览器离线打开，不依赖 SSH 或服务。原始数据、评分和 notebook 不受服务重启影响。

### 导出产物

导出目录位于挂载盘 `/mnt/tianzhu_hot/data-picking/reviews/math-web-retrieval-v2-4096-seed20261023/`：

- `index.html`：完整候选池的自包含审阅页。
- `math.samples.txt`、`background.samples.txt`、`openstax.samples.txt`、`finemath.samples.txt`：四类来源各固定随机 10 篇的完整正文及评分，方便在 VS Code 当普通文本阅读。不是按 loss 或主题挑选；`math.samples.txt` 仍仅包含 OpenWebMath。
- `review_manifest.json`：输入路径、来源计数和随机 seed。

生成器：[export_web_review.py](../scripts/maintenance/export_web_review.py)。在仓库根目录使用已有项目虚拟环境运行该脚本即可重新导出；默认包含四类来源，`--original-only` 可仅导出原两类。`--help` 列出原网页 / 新来源的正文目录、评分文件、输出目录及随机 seed 参数。

正文表与来源清单、评分表严格按 `id` 一对一关联，检查 ID 集合及来源标签一致；校验评分快照 SHA-256 和新增正文包完成清单，核对质量复验中两份背景与原背景分数完全一致后，只添加新来源的文档。不修改原始数据、评分或 notebook。原始输入说明见[网页数据准备](web_retrieval_data.md)和[数学文档质量复验](document_quality.md)。

保留原始 `source` 标签（`math` / `background`）；页面按 `corpus` 区分 `openwebmath` / `openstax` / `finemath` / `background`。教材额外展示标题、书名、章节和公式节点统计，FineMath 保留上游质量分数。

**来源不是语义标签：** `math` 表示各实验注入的数学相关来源，不是人工逐篇核验；`background` 表示未标注 FineWeb，不能将背景一概判断为非数学。教材图像仅保留文字描述，FineMath 高分也不保证正文和公式完整。页面上的查看行为不自动保存为人工标注。