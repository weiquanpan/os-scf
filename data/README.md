# 公开数据下载与准备

从仓库根目录运行以下命令。原始数据与生成的数据缓存不属于实验结果；只有小型 BikeSharing 原始文件随仓库提供。原始文件的大小按本地文件测量。

| 数据集 | 是否包含 | 本地原始数据大小 | 来源 |
| --- | --- | --- | --- |
| BikeSharing | 包含 `hour.csv`、`day.csv`、原始 `Readme.txt` | 合计约 1.16 MiB | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| LendingClub | 不包含 | 解压后约 1.56 GiB | [Kaggle: All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| NYC Taxi | 不包含 | 2024 年 1 月约 47.65 MiB | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |

## BikeSharing

随附 UCI 原始说明与引用要求。数据作者 Hadi Fanaee-T；相关论文为 Hadi Fanaee-T 与 João Gama 的 *Event labeling combining ensemble detectors and background knowledge*。来源与具体字段说明见同目录 `bike_sharing/Readme.txt`。

```bash
python -m experiments.preprocess_bikesharing --input data/bike_sharing/hour.csv --output-dir data/bike_sharing_preprocessed
```

生成 `data/bike_sharing_preprocessed/train.csv` 等文件，协议随后在这个开发池上执行分组划分。

## LendingClub

1. 打开上方 Kaggle 数据集页面，下载 `accepted_2007_to_2018Q4.csv.gz`（可能需要登录 Kaggle）。不需要 rejected 数据。
2. 解压 gzip 文件，将 CSV 保存为 `data/accepted_2007_to_2018Q4.csv`。
3. 显式传入文件名，避免 Linux 上 `.csv` / `.CSV` 大小写差异：

```bash
python -m experiments.preprocess_lendingclub --input data/accepted_2007_to_2018Q4.csv --output-dir data/lendingclub_preprocessed
```

协议读取 `data/lendingclub_preprocessed/train.csv`。预处理默认种子为 13，最多使用 200,000 行，训练比例为 0.8。勿替换成其他年份或经过不同筛选的同名数据集。

## NYC Taxi

在 TLC 官方页面展开 **2024 → January → Yellow Taxi Trip Records (PARQUET)**，下载 `yellow_tripdata_2024-01.parquet`，保存到：

```text
data/nyc_taxi_neuro/yellow_tripdata_2024-01.parquet
```

协议的 `prepare` 阶段会读取 Parquet，执行质量域筛选和确定性抽样，无需额外手动改列。该版本只使用上述月份的 Yellow Taxi 数据。

## 运行时生成

合成数据由 `experiments/neuro_mechanism_p2.py` 等脚本按固定配置生成，不需要下载。处理后数据、划分、采样结果、检查点和报告均被 Git 忽略；本仓库未包含实验结果。
