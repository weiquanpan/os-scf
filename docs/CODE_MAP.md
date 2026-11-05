# 代码入口索引

| 模块 | 用途 |
| --- | --- |
| `constrained_tabular_fm/continuous_original_space.py` | 原始空间自由坐标与归一化 |
| `constrained_tabular_fm/lendingclub_methods.py` | LendingClub 结构关系与方法 |
| `constrained_tabular_fm/bikesharing_methods.py` | BikeSharing 结构关系与方法 |
| `experiments/neurocomputing/common.py` | 数据、划分、codec、训练公共实现和溯源 |
| `experiments/neurocomputing/models.py` | 上游 TabSyn / TabDDPM 适配器 |
| `experiments/neurocomputing/run.py` | 主真实数据协议入口 |
| `experiments/neurocomputing/test_protocol.py` | 现有协议单元测试 |
| `experiments/neuro_mechanism_p2.py` | 合成数据冗余与光滑性机制实验 |
| `experiments/neuro_mechanism_p2_fast.py` / `p2_resume.py` / `p2_parallel.py` | P2 执行、恢复与并行支持 |
| `experiments/neuro_mechanism_p0.py` | 扩展验证选择敏感性 |
| `experiments/neuro_mechanism_real.py` / `stage.py` | 真实数据机制与分阶段执行 |
| `experiments/neuro_mechanism_cost_*.py` | 成本、有效性与质量轨迹 |
| `experiments/neurocomputing_redundancy_*.py` | 真实数据冗余先导与预算复查 |
| `experiments/neurocomputing_c1_diagnostics.py` | 保存样本的后验诊断 |
| `experiments/neurocomputing_*audit.py` | 布局、原始类别与检查点重放审计 |
| `experiments/report_neurocomputing_*.py` / `neuro_mechanism_report.py` | 从本地产物生成报告；报告本身不上传 |
| `experiments/run_constrained_tabular_flux.py` | 历史 Flux toy 示例，不是当前 OS-SCF 主实验 |

保留包结构以兼容脚本间引用，不改变模型或指标实现。`source-manifest.json` 记录实际复制的代码和小型数据校验值，不将原仓库 HEAD 误当成包含全部未提交修改的完整版本。
