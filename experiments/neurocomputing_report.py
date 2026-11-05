"""Generate submission-oriented tables from a completed, audited experiment.

This post-hoc report never chooses checkpoints or changes training budgets.
Split variability is descriptive: overlapping random splits are not independent
datasets and five seeds in one split are not five independent test sets.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["osscf", "direct", "direct_repair", "tabsyn", "tabsyn_repair",
           "free_tabsyn", "tabddpm", "train_bootstrap"]
LABELS = {"osscf": "OS-SCF", "direct": "Direct", "direct_repair": "Direct+Phi",
          "tabsyn": "TabSyn", "tabsyn_repair": "TabSyn+Phi",
          "free_tabsyn": "Free-TabSyn", "tabddpm": "TabDDPM",
          "train_bootstrap": "Train bootstrap", "real_matched": "Real matched"}
DATASETS = {"lendingclub": "LendingClub", "bikesharing": "BikeSharing",
            "taxi": "NYC Taxi"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_statistics(frame, groups, values):
    """Equal weight per split after averaging seeds inside each split."""
    first = frame.groupby(groups + ["split"], dropna=False)[values].mean()
    tidy = first.reset_index().melt(id_vars=groups + ["split"],
                                   var_name="metric", value_name="value")
    return tidy.groupby(groups + ["metric"], dropna=False).value.agg(
        mean="mean", split_sd="std", n_splits="count").reset_index()


def latex_table(path, frame, caption, label):
    text = frame.to_latex(index=False, escape=True, na_rep="--",
                          caption=caption, label=label, position="htbp")
    text = text.replace(r"\begin{table}[htbp]", r"\begin{table*}[t]" + "\n" + r"\centering")
    text = text.replace(r"\begin{tabular}", r"\resizebox{\textwidth}{!}{%" + "\n" + r"\begin{tabular}")
    text = text.replace(r"\end{tabular}", r"\end{tabular}" + "\n}")
    text = text.replace(r"\end{table}", r"\end{table*}")
    path.write_text(text, encoding="utf8")


def formatted(stats, dataset, method, metric, model=None):
    row = stats[(stats.dataset == dataset) & (stats.method == method)
                & (stats.metric == metric)]
    if model is not None:
        row = row[row.utility_model == model]
    if len(row) != 1 or row.iloc[0].n_splits == 0:
        return "--"
    r = row.iloc[0]
    sd = f"{r.split_sd:.4f}" if np.isfinite(r.split_sd) else "--"
    return f"{r['mean']:.4f} +/- {sd}"


def make_report(out):
    out = Path(out)
    audit = json.loads((out / "audit/report.json").read_text())
    if not (out / "COMPLETE.json").exists() or audit["status"] != "PASS":
        raise RuntimeError("Only a complete, audited protocol may produce the final report")
    layout_audit = json.loads((out / "audit/layout_replay.json").read_text())
    if layout_audit["status"] != "PASS" or not layout_audit["evidence_unchanged"]:
        raise RuntimeError("Canonical-layout replay and evidence preservation must pass")
    if layout_audit["source_sha256"] != digest(Path(__file__).with_name("neurocomputing_layout_audit.py")):
        raise RuntimeError("Layout-audit source changed: rerun layout replay before reporting")
    replay = json.loads((out / "replay_audit/report.json").read_text())
    if replay["status"] != "PASS":
        raise RuntimeError("The additional ablation and checkpoint replay must pass first")
    config = json.loads((out / "protocol.json").read_text())
    category_manifest = json.loads((out / "raw_category_audit/manifest.json").read_text())
    if not category_manifest["complete_main_protocol"]:
        raise RuntimeError("Rerun the raw-category evaluator after the main audit")
    if replay["source_sha256"] != digest(Path(__file__).with_name("neurocomputing_replay_audit.py")):
        raise RuntimeError("Replay-audit source changed: rerun the replay audit before reporting")
    if category_manifest["source_sha256"] != digest(Path(__file__).with_name("neurocomputing_raw_category_audit.py")):
        raise RuntimeError("Categorical-audit source changed: rerun categorical diagnostics before reporting")
    inputs = [out / p for p in ["summary/metrics_per_run.csv", "summary/utility_per_run.csv",
              "summary/ablation_per_run.csv", "raw_category_audit/per_run.csv",
              "audit/report.json", "audit/tabsyn_budget_flags.csv", "protocol.json", "replay_audit/report.json",
              "audit/layout_replay.json", "audit/precision_sensitivity.csv"]]
    metrics, utility, ablation, categories = [pd.read_csv(p) for p in inputs[:4]]
    keys = ["dataset", "split", "seed", "method"]
    if metrics.duplicated(keys).any() or categories.duplicated(keys).any():
        raise ValueError("Duplicate experiment keys")
    merged = metrics.merge(categories, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Raw-category diagnostics do not cover the same experiment keys")
    n_seeds, n_splits = len(config["training_seeds"]), len(config["split_seeds"])
    expected = n_seeds * n_splits * sum(8 + (4 if d != "taxi" else 0) for d in config["datasets"])
    expected_ablation = n_seeds * n_splits * sum(d != "taxi" for d in config["datasets"]) * len(config["flow"]["alphas"]) * 3
    if len(metrics) != expected or len(utility) != 2 * expected or len(ablation) != expected_ablation:
        raise ValueError("Unexpected coverage of final, utility, or ablation rows")
    if not utility.status.eq("ok").all():
        raise ValueError("A utility fit failed")
    for frame, groups in [(metrics, ["dataset", "split", "method"]),
                          (utility, ["dataset", "split", "method", "utility_model"]),
                          (ablation, ["dataset", "split", "method"])]:
        if not frame.groupby(groups).size().eq(n_seeds).all():
            raise ValueError("Incomplete training-seed coverage")

    dest = out / "report"
    dest.mkdir(exist_ok=True)
    reference_rows = []
    for ds in config["datasets"]:
        for split in config["split_seeds"]:
            path = out / f"split{split}" / "data" / ds / "reference_audit.json"
            record = json.loads(path.read_text())
            inputs.append(path)
            reference_rows.append({"dataset": ds, "split": split,
                                   "test_rows": record["test_rows"],
                                   **{k: v for k, v in record["test_reference_structure"].items()
                                      if "violation" in k or "unknown" in k or k == "constraint_mae"}})
    reference = pd.DataFrame(reference_rows)
    reference.to_csv(dest / "real_test_structure.csv", index=False)
    fidelity_values = ["sw", "free_sw", "dep_sw", "violation_rate", "constraint_mae",
                       "pearson_mae", "spearman_mae", "category_tv",
                       "raw_marginal_category_tv", "raw_pairwise_category_tv", "raw_joint_category_tv",
                       "generated_unknown_rate", "reference_unknown_rate", "negative_violation",
                       "integer_violation", "fico_grid_violation", "logic_violation"]
    fidelity_values = [v for v in fidelity_values if v in merged]
    fidelity = split_statistics(merged, ["dataset", "method"], fidelity_values)
    fidelity.to_csv(dest / "fidelity_split_statistics.csv", index=False)
    utility_stats = split_statistics(utility, ["dataset", "method", "utility_model"],
                                     [c for c in ["roc_auc", "average_precision", "r2", "rmse", "mae"] if c in utility])
    utility_stats.to_csv(dest / "utility_split_statistics.csv", index=False)
    ablation = ablation.copy()
    ablation["base_method"] = ablation.method.str.replace(r"_alpha.*$", "", regex=True)
    abl_stats = split_statistics(ablation, ["dataset", "base_method", "alpha"], ["sw", "free_sw", "dep_sw"])
    abl_stats.to_csv(dest / "ablation_split_statistics.csv", index=False)

    for ds in config["datasets"]:
        table = []
        for method in METHODS:
            table.append({"Method": LABELS[method], **{
                label: formatted(fidelity, ds, method, metric)
                for label, metric in [("Full SW", "sw"), ("Free SW", "free_sw"),
                                      ("Equation violation", "violation_rate"),
                                      ("Pairwise category TV", "raw_pairwise_category_tv")]}})
        latex_table(dest / f"fidelity_{ds}.tex", pd.DataFrame(table),
                    f"{DATASETS[ds]}: mean and descriptive SD of split means; {n_seeds} seeds per split. SW covers numerical coordinates standardized using training statistics; categories are evaluated separately. All metrics are lower-is-better. Train bootstrap is a reference, not a fitted competitor.",
                    f"tab:nc-fidelity-{ds}")
    rows = []
    for ds in config["datasets"]:
        metric = "roc_auc" if ds == "lendingclub" else "r2"
        for method in METHODS[:-1] + ["real_matched"]:
            rows.append({"Dataset": DATASETS[ds], "Method": LABELS[method], "Metric": metric,
                         "Random forest": formatted(utility_stats, ds, method, metric, "rf"),
                         "Linear": formatted(utility_stats, ds, method, metric, "linear")})
    latex_table(dest / "utility.tex", pd.DataFrame(rows),
                "TSTR: mean and descriptive SD of split means. Larger is better. Matched real training data use the same number of rows as synthetic training data. Target operands are excluded for BikeSharing and Taxi.",
                "tab:nc-utility")
    rows = []
    for ds in [d for d in config["datasets"] if d != "taxi"]:
        for base in ["osscf", "direct_repair", "tabsyn_repair", "free_tabsyn"]:
            for method in [base, base + "_rounded"]:
                rows.append({"Dataset": DATASETS[ds], "Method": LABELS[base] + (" rounded" if method != base else ""),
                             **{label: formatted(fidelity, ds, method, metric)
                                for label, metric in [("Full SW", "sw"), ("Integer violation", "integer_violation"),
                                                      ("Equation violation", "violation_rate")]}})
    latex_table(dest / "rounding.tex", pd.DataFrame(rows),
                "Lattice projection sensitivity: free variables are rounded before dependent variables are reconstructed. Means and descriptive SDs are across split means.",
                "tab:nc-rounding")

    pairs = []
    base = metrics[metrics.method == "osscf"].set_index(["dataset", "split", "seed"])
    comparisons = ["direct_repair", "tabsyn_repair", "free_tabsyn", "tabddpm"]
    for method in comparisons:
        other = metrics[metrics.method == method].set_index(["dataset", "split", "seed"])
        difference = (base.sw - other.sw).rename("difference").reset_index()
        for (ds, split), g in difference.groupby(["dataset", "split"]):
            pairs.append({"dataset": ds, "split": split, "comparison": method,
                          "mean_difference": g.difference.mean(), "seed_sd": g.difference.std(),
                          "n_seeds": len(g), "negative_seed_count": int(g.difference.lt(0).sum())})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(dest / "paired_sw_differences.csv", index=False)
    fig, axes = plt.subplots(1, len(config["datasets"]), figsize=(15, 4), squeeze=False)
    colors = ["#355c9d", "#b35b16", "#288073"]
    for ax, ds in zip(axes[0], config["datasets"]):
        for i, split in enumerate(config["split_seeds"]):
            g = pairs[(pairs.dataset == ds) & (pairs.split == split)].set_index("comparison").loc[comparisons]
            offset = (i - (n_splits - 1) / 2) * .19
            ax.errorbar(g.mean_difference, np.arange(len(comparisons)) + offset,
                        xerr=g.seed_sd.fillna(0), fmt="o", markersize=4, capsize=2,
                        color=colors[i % len(colors)], label=f"Split {split}")
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_yticks(range(len(comparisons)), [LABELS[m] for m in comparisons])
        ax.set_title(DATASETS[ds]); ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
        ax.set_xlabel("OS-SCF SW minus comparator SW\nnegative favors OS-SCF; bars = seed SD")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(dest / "paired_sw.pdf"); fig.savefig(dest / "paired_sw.png", dpi=180); plt.close(fig)

    # Preserve the all-comparator plot, with a separate view for the three
    # reconstruction-based comparators whose smaller differences need detail.
    detail_comparisons = comparisons[:3]
    fig, axes = plt.subplots(1, len(config["datasets"]), figsize=(15, 4.4), squeeze=False)
    for ax, ds in zip(axes[0], config["datasets"]):
        for i, split in enumerate(config["split_seeds"]):
            g = pairs[(pairs.dataset == ds) & (pairs.split == split)].set_index("comparison").loc[detail_comparisons]
            offset = (i - (n_splits - 1) / 2) * .19
            ax.errorbar(g.mean_difference, np.arange(len(detail_comparisons)) + offset,
                        xerr=g.seed_sd.fillna(0), fmt="o", markersize=4, capsize=2,
                        color=colors[i % len(colors)], label=f"Split {split}")
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_yticks(range(len(detail_comparisons)), [LABELS[m] for m in detail_comparisons])
        ax.set_title(DATASETS[ds]); ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
        ax.set_xlabel("OS-SCF SW minus comparator SW\nnegative favors OS-SCF; bars = seed SD")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Detail: reconstruction-based comparators (all comparators shown in the overview)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(dest / "paired_sw_reconstruction_detail.pdf")
    fig.savefig(dest / "paired_sw_reconstruction_detail.png", dpi=180)
    plt.close(fig)

    utility_pairs = []
    for ds in config["datasets"]:
        metric = "roc_auc" if ds == "lendingclub" else "r2"
        for model in ["rf", "linear"]:
            frame = utility[(utility.dataset == ds) & (utility.utility_model == model)]
            target = frame[frame.method == "osscf"].set_index(["split", "seed"])[metric]
            for method in comparisons + ["real_matched"]:
                other = frame[frame.method == method].set_index(["split", "seed"])[metric]
                difference = (target - other).rename("difference").reset_index()
                for split, group in difference.groupby("split"):
                    utility_pairs.append({"dataset": ds, "utility_model": model, "metric": metric,
                                          "split": split, "comparison": method,
                                          "mean_difference": group.difference.mean(),
                                          "seed_sd": group.difference.std(), "n_seeds": len(group),
                                          "positive_seed_count": int(group.difference.gt(0).sum())})
    utility_pairs = pd.DataFrame(utility_pairs)
    utility_pairs.to_csv(dest / "paired_utility_differences.csv", index=False)

    abl_ds = [d for d in config["datasets"] if d != "taxi"]
    fig, axes = plt.subplots(1, len(abl_ds), figsize=(10, 4.2), squeeze=False)
    for ax, ds in zip(axes[0], abl_ds):
        for method, color in [("osscf", "#5754c6"), ("direct_repair", "#b35b16")]:
            g = abl_stats[(abl_stats.dataset == ds) & (abl_stats.base_method == method) & (abl_stats.metric == "sw")].sort_values("alpha")
            ax.errorbar(g.alpha, g["mean"], yerr=g.split_sd.fillna(0), fmt="o-", capsize=4,
                        color=color, label=LABELS[method])
        ax.set_title(DATASETS[ds]); ax.set_xlabel("Shrinkage alpha"); ax.set_ylabel("Full SW (bars = split SD)")
        ax.set_xticks(config["flow"]["alphas"]); ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(dest / "alpha_ablation.pdf"); fig.savefig(dest / "alpha_ablation.png", dpi=180); plt.close(fig)

    caps = pd.read_csv(inputs[5])
    cap_count = int((caps.vae_hit_cap | caps.diffusion_hit_cap).sum())
    # Fixed first split/seed, also used by the saved-model replay. Do not choose
    # attractive-looking learning curves after inspecting the test results.
    first_split, first_seed = config["split_seeds"][0], config["training_seeds"][0]
    fig, axes = plt.subplots(len(config["datasets"]), 2,
                             figsize=(10, 3 * len(config["datasets"])), squeeze=False)
    for row_index, ds in enumerate(config["datasets"]):
        for method, color in [("tabsyn", "#355c9d"), ("free_tabsyn", "#b35b16")]:
            model_folder = out / f"split{first_split}" / "final" / ds / f"seed{first_seed}" / method
            info_path = model_folder / "trained.json"
            info = json.loads(info_path.read_text())
            inputs.append(info_path)
            for column, (stage, metric) in enumerate([("vae", "validation_reconstruction"),
                                                      ("diffusion", "validation_sw")]):
                history_path = model_folder / f"{stage}_history.csv"
                history = pd.read_csv(history_path).dropna(subset=[metric])
                inputs.append(history_path)
                ax = axes[row_index, column]
                ax.plot(history.epoch, history[metric], color=color, label=LABELS[method], linewidth=1.4)
                selected = history[history.epoch == info[f"{stage}_selected_epoch"]]
                ax.scatter(selected.epoch, selected[metric], color=color, s=28, zorder=3)
                ax.set_title(f"{DATASETS[ds]}: {stage.upper() if stage == 'vae' else 'diffusion'}")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Validation reconstruction" if column == 0 else "Validation full SW")
                if column == 0:
                    ax.set_yscale("log")
                ax.grid(alpha=.18); ax.legend(fontsize=8)
    fig.suptitle(f"Fixed split {first_split}, seed {first_seed}; dots mark selected checkpoints", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(dest / "baseline_learning_curves.pdf")
    fig.savefig(dest / "baseline_learning_curves.png", dpi=180)
    plt.close(fig)
    title = "SMOKE TEST ONLY — NOT SCIENTIFIC RESULTS" if config.get("smoke_only") else "Neurocomputing 补充实验结果"
    lines = [title, "", f"主评价 {len(metrics)} 行；下游任务 {len(utility)} 行；消融评价 {len(ablation)} 行。",
             f"证据一致性审计：{audit['status']}；最大指标重算误差 {audit['max_replay_error']:.3g}。",
             f"补充重放审计：{replay['status']}；{replay['ablation_metric_rows']} 行消融指标和 {replay['saved_model_checks']} 个固定检查点已重算。",
             f"每数据集 {n_splits} 次随机划分，每划分 {n_seeds} 个训练种子。先计算种子均值，再计算划分均值和标准差。", "",
             "全部数值维度 SW 的描述性结果（类别另行评估；不是显著性结论）："]
    for ds in config["datasets"]:
        g = fidelity[(fidelity.dataset == ds) & (fidelity.metric == "sw") & fidelity.method.isin(METHODS[:-1])].sort_values("mean")
        winner = g.iloc[0]
        lines.append(f"{DATASETS[ds]}：均值最低为 {LABELS[winner.method]}（{winner['mean']:.4f}）；OS-SCF 为 {formatted(fidelity, ds, 'osscf', 'sw')}。")
        for method in comparisons:
            delta = pairs[(pairs.dataset == ds) & (pairs.comparison == method)].mean_difference
            lines.append(f"  对 {LABELS[method]}：OS-SCF 在 {int(delta.lt(0).sum())}/{len(delta)} 个划分的配对平均 SW 更低；平均差 {delta.mean():+.5f}。")
    lines += ["", "下游效用（数值越大越好；均值与描述性划分标准差）："]
    for ds in config["datasets"]:
        metric = "roc_auc" if ds == "lendingclub" else "r2"
        for model in ["rf", "linear"]:
            lines.append(f"{DATASETS[ds]} / {model} / {metric}：OS-SCF {formatted(utility_stats, ds, 'osscf', metric, model)}；Direct+Phi {formatted(utility_stats, ds, 'direct_repair', metric, model)}；匹配真实数据 {formatted(utility_stats, ds, 'real_matched', metric, model)}。")
    lines += ["完整基线结果见 utility.tex 和 utility_split_statistics.csv；逐划分配对差值见 paired_utility_differences.csv。微小均值差异不等于显著优势。",
              "", "固定训练步数的 alpha 消融（仅作事后描述，不据此更换主实验配置）：",
              "alpha=0 为完整 ZCA 白化，alpha=1 为向单位阵完全收缩、近似仅标准化。"]
    for ds in abl_ds:
        for method in ["osscf", "direct_repair"]:
            g = abl_stats[(abl_stats.dataset == ds) & (abl_stats.base_method == method) & (abl_stats.metric == "sw")].sort_values("mean")
            best = g.iloc[0]
            lines.append(f"{DATASETS[ds]} / {LABELS[method]}：观察到最低平均 SW 的 alpha 为 {best.alpha:g}，SW={best['mean']:.5f}。")
    lines += ["消融结果不支持预先断言中间收缩系数一定优于两个端点；均值次序也不构成统计显著性结论。",
              "", "真实测试记录的等式违规率（各划分均值；不是生成结果）："]
    for ds, value in reference.groupby("dataset").violation_rate.mean().items():
        lines.append(f"{DATASETS[ds]}：{value:.3%}。")
    hard = metrics[metrics.method.str.replace(r"_rounded$", "", regex=True).isin(
        ["osscf", "direct_repair", "tabsyn_repair", "free_tabsyn"])]
    rounded = hard[hard.method.str.endswith("_rounded")]
    lines += ["", "设定约束的逐次核验（以下均为所有相关运行中的最大违规率）：",
              f"硬重构方法的数值等式：{hard.violation_rate.max():.3%}。",
              f"BikeSharing 工作日逻辑：{hard.loc[hard.dataset == 'bikesharing', 'logic_violation'].max():.3%}。",
              f"取整变体的整数约束：{rounded.integer_violation.max():.3%}；LendingClub FICO 网格：{rounded.loc[rounded.dataset == 'lendingclub', 'fico_grid_violation'].max():.3%}。",
              "这些检查只覆盖已声明的等式、逻辑和网格规则，并不表示满足全部现实业务约束。"]
    lines += ["", "适用范围与限制：",
              f"审计读回 CSV 时恢复生成张量的行连续布局，避免 FP32 求和顺序改变阈值边界判断；原始样本和指标文件均未修改。另对全部 {layout_audit['precision_sensitivity_rows']} 份主评价做了 FP64 敏感性核查，违规率最大差值为 {100 * layout_audit['max_fp32_fp64_rate_difference']:.3f} 个百分点；硬重构方法的 FP64 最大等式违规率为 {layout_audit['fp64_hard_equation_max_violation_rate']:.3%}。详见 audit/precision_sensitivity.csv。",
              f"TabSyn/Free-TabSyn 共 {len(caps)} 次拟合，其中 {cap_count} 次至少一个阶段达到预算上限。这里只实现了带预算上限、验证集选模的上游组件适配器，不宣称精确复现官方完整训练或已收敛。",
              "TabDDPM 使用上游训练与采样实现，但限于 128×128 MLP、100 个扩散时间步，训练步数从 3000/10000 中按验证集选择。这是明确预算下的配置，并非逐数据集充分调参的强复现；较差结果不能用来否定 TabDDPM 方法本身。",
              "三个随机划分来自同一数据池且会相互重叠。这里的标准差是描述性稳定性指标，不是基于独立数据集的置信区间，也未把 15 个种子当成 15 个独立测试集。",
              "LendingClub/BikeSharing 数据池在历史开发中已使用；它们不是全新的外部验证集。",
              "Taxi 使用 2024 年 1 月数据的供应商相关经验记账规则；原始数据仍有违反记录。结构重建不等于恢复所有现实噪声或覆盖所有业务约束。",
              "类别指标额外保留原始标签，报告边际、两两联合和全联合 TV。高维联合 TV 强烈依赖样本量，需与同样本量的 bootstrap 参考一起看。",
              "Flow 条件类别从训练数据经验分布抽样；Free-TabSyn 则学习生成类别。应同时报告类别联合分布与下游效用，不能只凭数值 SW 宣称全面更好。",
              "Direct 的原始输出与 Direct+Phi 共用验证选择；原始 Direct 是诊断控制，不是单独充分调参的无约束强基线。",
              "所有模型均遵守相同 train/val/test 划分，但参数规模、训练预算及选择成本不完全一致；计时也不保证独占显卡，不作严格公平加速比结论。",
              "报告中的非零、负向或不支持方法优越性的结果均保留。此次程序完成和审计通过不等同于论文已达到期刊录用要求。"]
    (dest / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    manifest = {"report_source_sha256": digest(__file__), "inputs": {str(p.relative_to(out)): digest(p) for p in inputs},
                "metric_rows": len(metrics), "utility_rows": len(utility), "ablation_rows": len(ablation),
                "interpretation": "descriptive split variability, no independent-split significance claims"}
    source_folder = dest / "source_snapshot"
    source_folder.mkdir(exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("neurocomputing_raw_category_audit.py"),
                    Path(__file__).with_name("neurocomputing_replay_audit.py"),
                    Path(__file__).with_name("neurocomputing_layout_audit.py"),
                    Path(__file__).with_name("neurocomputing_complete.py")]
    for path in source_paths:
        shutil.copy2(path, source_folder / path.name)
    manifest["posthoc_sources"] = {p.name: digest(p) for p in source_paths}
    manifest["outputs"] = {p.name: digest(p) for p in dest.iterdir() if p.is_file() and p.name != "manifest.json"}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps({"report": str(dest), **{k: manifest[k] for k in ["metric_rows", "utility_rows", "ablation_rows"]}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    make_report(args.out)
