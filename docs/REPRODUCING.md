# 运行与复现

## 上游基线

基线代码不复制进仓库，保留其原始许可。以下提交来自本地实验所用仓库的实际 HEAD。

```bash
git clone https://github.com/amazon-science/tabsyn.git external/tabsyn
git -C external/tabsyn checkout cb5ac0f74ec36ee88e7a974a393dfbef50d42da7
git clone https://github.com/yandex-research/tab-ddpm.git external/tab-ddpm
git -C external/tab-ddpm checkout b476257dd460b778ba09eb97f7a51d6490fa17f8
```

安装 `requirements.txt` 后，按所用基线安装额外依赖；TabDDPM 的原始依赖文件含较老的固定版本，不宜直接覆盖主环境。`requirements-baselines.txt` 记录本机已安装的相关额外依赖版本；完整隔离环境安装尚未验证。主协议适配器包含 sklearn integer-subsample 兼容处理。

## 快速检查

```bash
python scripts/verify_snapshot.py
python -m experiments.neurocomputing.test_protocol
python -m experiments.preprocess_bikesharing --input data/bike_sharing/hour.csv --output-dir data/bike_sharing_preprocessed
python -m experiments.neurocomputing.run --out outputs/bike_smoke --smoke --datasets bikesharing --stage flow
```

最后一步需要 TabSyn 源码及其导入依赖，即使没有训练 TabSyn。烟雾配置缩小样本量和迭代次数，不用于论文数值。

## 主协议

完成 `data/README.md` 中三个数据集准备和上游依赖安装后：

```bash
python -m experiments.neurocomputing.run --out outputs/neurocomputing_v2 --stage all
python -m experiments.neurocomputing_layout_audit --out outputs/neurocomputing_v2
python -m experiments.neurocomputing_raw_category_audit --out outputs/neurocomputing_v2
python -m experiments.neurocomputing_replay_audit --out outputs/neurocomputing_v2
python -m experiments.neurocomputing_report --out outputs/neurocomputing_v2
```

配置在 `experiments/neurocomputing/protocol.json`。源代码会冻结输出目录的协议及源码身份；改变训练代码后使用新输出目录。后续机制脚本默认读取 `outputs/neurocomputing_v2` 及其中的 `provenance.json`，因此不能从一个空输出目录直接运行全部机制实验。

## 合成与后续诊断

先查看 `docs/CODE_MAP.md` 和对应脚本的参数、前置输入。P2 的 `--check-only` 可以检查合成机制配置；完整网格扩展受原脚本预设的 screen decision 控制。保留这些门槛，不以跳过校验来获得完成标记。

历史 `neurocomputing_complete.py` / 报告协调器还包含归档旧稿的路径假设。仓库不提供旧论文或历史输出；以上直接阶段命令是代码发布的运行入口。论文排版和稿件生成脚本未纳入发布。
