# GDN 紧凑状态复核证据

本目录只对应 2026-10-20 的 GDN 复核；共享规划、变长 query attention 筛选保持结束。

- `lifecycle-v1.json`、`lifecycle-v2.json`：两个独立进程的真实单层原始结果。
- `analysis.json`：由 `benchmarks/multi_graph_gdn_review_analyze.py` 重算的逐轮比较。
- `frozen-protocol.json`：输入、接受长度、几何及计时边界。
- `installed-manifest.json`：安装源码、包版本及实际使用库的 SHA。
- `deployed-sources.json`：v2 上传脚本的规范化 LF 源码 SHA。
- `measured-v1-layer-probe.py`：首轮被测层探针；v2 仅修正报告边界说明及打印。
- `community-source-manifest.json`：固定社区提交、源码链接及 SHA。
- `source-snapshots.tar.gz`：原样保存的社区/安装源码及两轮日志，不进行项目风格重写；解包后恢复 `community/`、`installed/` 等相对路径。
- `snapshot-archive-manifest.json`：归档及每个成员的 SHA；`community/` 下的 PR 元数据另外保留为 JSON。
- `validation.json`、`format-ci.log`：检查结果及本机格式 CI 阻断原因。

外部源码保留其原始版权与许可标记；Apache 2.0 及 CANN 许可文本位于归档内的 `community/LICENSES/`。
外部代码和 PR 尚未合并的状态均不意味着它们不是已有工作。

结论：保留 A2 ready/提交调度为有条件主线候选；不把紧凑状态表示称为新算法。
尚未实测同 A2 的社区紧凑状态基线，不声称优于社区最佳、模型或服务吞吐提升。
