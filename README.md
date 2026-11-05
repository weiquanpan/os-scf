# OS-SCF

Original-Space Structural-Coordinate Constrained Flow：在自由坐标上学习流模型，再利用已知结构关系重建依赖列。

本仓库是以 **paper-v10 为基准的代码快照**。包含核心模型、对应实验协议、诊断与审计脚本、小型公开数据；不包含论文、实验结果、模型权重或旧稿。保留原始模块路径及训练代码字节内容，避免破坏导入和实验溯源校验。

## 目录

| 路径 | 内容 |
| --- | --- |
| `constrained_tabular_fm/` | 模型、结构约束、训练、采样及指标 |
| `experiments/neurocomputing/` | 真实数据实验协议、基线适配器、评估与现有测试 |
| `experiments/neuro_mechanism_*.py` | 合成冗余实验、机制分析和成本诊断 |
| `experiments/neurocomputing_*.py` | 真实数据诊断、复核及实验协调脚本 |
| `experiments/preprocess_*.py` | BikeSharing / LendingClub 数据预处理 |
| `data/bike_sharing/` | 小型公开 BikeSharing 原始数据，约 1.16 MiB |
| `docs/` | 运行说明、代码索引、源文件校验与环境记录 |

## 安装和检查

在仓库根目录执行。建议使用独立环境；本次验证使用 Python 3.13 和 `docs/environment-snapshot.json` 记录的已安装依赖。依赖文件不是所有操作系统上的完整锁文件。

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
python -m experiments.neurocomputing.test_protocol
```

运行真实数据协议前，按照 [复现说明](docs/REPRODUCING.md) 获取上游 TabSyn / TabDDPM。当前协议模块在导入时加载 TabSyn，所以仅选择 flow 阶段也需要它。

## 数据

BikeSharing 原始数据已附带。生成实验所需输入：

```bash
python -m experiments.preprocess_bikesharing --input data/bike_sharing/hour.csv --output-dir data/bike_sharing_preprocessed
```

LendingClub 和 NYC Taxi 未上传。下载链接、文件名、保存位置、预处理命令见 [数据说明](data/README.md)。所有实验输出默认写入被 Git 忽略的 `outputs/`。

## 实验入口

- [复现说明](docs/REPRODUCING.md)：依赖、真实数据烟雾运行和正式协议。
- [代码索引](docs/CODE_MAP.md)：合成机制、真实数据诊断与历史示例的区别。
- [原始协议说明](experiments/neurocomputing/README.md)：划分、模型选择、基线预算与审计规则。

现有测试和烟雾运行用于检查整理后的执行链；不代表重新跑完论文实验或复核全部论文数值。复用已有输出时，源代码与协议必须通过其溯源检查。完整实验需要额外数据、上游依赖和相应计算资源。
