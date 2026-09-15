import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

RESULTS_COLUMNS = [
    "timestamp", "method", "dataset", "missing_rate",
    "trial", "seed", "n_neighbors",
    "accuracy", "tuning_time_sec", "train_time_sec", "predict_time_sec", "total_time_sec",
]

# màu cố định theo tên phương pháp, giữ nhất quán giữa các dataset/biểu đồ
METHOD_COLORS = {
    "KNN-IC": "tab:green",
    "Mean-IC": "tab:purple",
    "KNN-Direct": "tab:blue",
    "KNN-Direct-Weighted": "tab:orange",
    "KNN-Direct-Manhattan": "tab:red",
}


def run_trial(method_func, x_train_missing, x_test_missing, y_train, y_test, **method_kwargs):
    start = time.perf_counter()
    result = method_func(x_train_missing, x_test_missing, y_train, y_test, **method_kwargs)
    result["total_time_sec"] = time.perf_counter() - start
    return result


def save_result(result_row: dict, filepath: str):
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    df_row = pd.DataFrame([result_row])[RESULTS_COLUMNS]
    write_header = not os.path.exists(filepath)
    df_row.to_csv(filepath, mode="a", header=write_header, index=False)


def run_experiment(methods: dict, get_split_func, missing_rates: list,
                    n_trials: int, dataset_name: str, output_path: str,
                    base_seed: int = 0, overwrite: bool = True):
    """
    methods: {method_name: callable(x_train_missing, x_test_missing, y_train, y_test) -> dict}
    get_split_func: callable(seed) -> (x_train, x_test, y_train, y_test) đã chuẩn hóa, chưa missing
    overwrite: xóa output_path trước khi chạy (mặc định True) để tránh trộn dữ liệu cũ/mới
               từ những lần chạy trước đó, kể cả khi chạy với cấu hình khác (methods, MAX_TUNING_NEIGHBORS...).
    """
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
    for missing_rate in missing_rates:
        print("missing rate: ", missing_rate);
        for trial in range(n_trials):
            print("trial: ", trial);
            seed = base_seed + trial
            x_train, x_test, y_train, y_test = get_split_func(seed)

            rng = np.random.RandomState(seed)
            x_train_missing = generate_missing_values(x_train, missing_rate, rng)
            x_test_missing = generate_missing_values(x_test, missing_rate, rng)

            for method_name, method_func in methods.items():
                result = run_trial(method_func, x_train_missing, x_test_missing, y_train, y_test,
                                    random_state=seed)
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "method": method_name,
                    "dataset": dataset_name,
                    "missing_rate": missing_rate,
                    "trial": trial,
                    "seed": seed,
                    **result,
                }
                save_result(row, output_path)


def generate_missing_values(original_data, missing_rate, rng: np.random.RandomState):
    data_shape = original_data.shape
    n_missing = round(missing_rate * original_data.size)
    missing_id = rng.choice(original_data.size, n_missing, replace=False)
    missing_data = original_data.flatten().copy()
    missing_data[missing_id] = np.nan
    return missing_data.reshape(data_shape)


def load_results(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def summarize_results(df: pd.DataFrame, metrics=("accuracy", "total_time_sec", "n_neighbors")) -> pd.DataFrame:
    # luôn nhóm kèm "dataset" — vô hại khi df chỉ có 1 dataset, nhưng ngăn việc âm thầm
    # tính trung bình lẫn lộn giữa nhiều dataset nếu df là kết quả gộp từ nhiều file CSV.
    grouped = df.groupby(["dataset", "method", "missing_rate"])
    summary = grouped[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n"] = grouped.size().values
    for metric in metrics:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(summary["n"])
    if "n_neighbors" in metrics:
        # mode thường dễ đọc hơn mean cho 1 hyperparameter rời rạc (VD: "k=7 được chọn nhiều nhất")
        summary["n_neighbors_mode"] = grouped["n_neighbors"].agg(lambda s: s.mode().iloc[0]).values
    return summary.reset_index()


def compare_methods(df: pd.DataFrame, method_a: str, method_b: str,
                     missing_rate: float, metric: str = "accuracy") -> dict:
    if df["dataset"].nunique() > 1:
        raise ValueError(
            "df chứa nhiều dataset — paired test chỉ có ý nghĩa trong 1 dataset. "
            "Hãy lọc df theo dataset trước, hoặc dùng compare_methods_across_datasets()."
        )
    a = df[(df.method == method_a) & (df.missing_rate == missing_rate)].sort_values("trial")[metric]
    b = df[(df.method == method_b) & (df.missing_rate == missing_rate)].sort_values("trial")[metric]
    t_stat, t_p = stats.ttest_rel(a, b)
    w_stat, w_p = stats.wilcoxon(a, b) if len(a) >= 10 else (None, None)
    return {
        "missing_rate": missing_rate,
        "mean_a": a.mean(), "mean_b": b.mean(),
        "ttest_p_value": t_p,
        "wilcoxon_p_value": w_p,
    }


def compare_methods_across_datasets(dataset_dfs: dict, method_a: str, method_b: str,
                                     metric: str = "accuracy") -> pd.DataFrame:
    """
    dataset_dfs: {dataset_name: df} - chỉ những dataset có cả method_a và method_b mới được đưa vào bảng
    """
    rows = []
    for dataset_name, df in dataset_dfs.items():
        methods_present = set(df["method"].unique())
        if method_a not in methods_present or method_b not in methods_present:
            continue
        for missing_rate in sorted(df["missing_rate"].unique()):
            cmp = compare_methods(df, method_a, method_b, missing_rate=missing_rate, metric=metric)
            rows.append({
                "dataset": dataset_name,
                "missing_rate": missing_rate,
                method_a: round(cmp["mean_a"], 4),
                method_b: round(cmp["mean_b"], 4),
                "diff": round(cmp["mean_b"] - cmp["mean_a"], 4),
                "ttest_p": cmp["ttest_p_value"],
                "wilcoxon_p": cmp["wilcoxon_p_value"],
            })
    return pd.DataFrame(rows)


def plot_methods_across_datasets(dataset_dfs: dict, method_a: str, method_b: str,
                                  metric: str = "accuracy", output_path: str = None):
    import matplotlib.pyplot as plt

    valid = {name: df for name, df in dataset_dfs.items()
             if method_a in df["method"].unique() and method_b in df["method"].unique()}
    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4), squeeze=False)
    for ax, (dataset_name, df) in zip(axes[0], valid.items()):
        summary = summarize_results(df, metrics=(metric,))
        for method_name in (method_a, method_b):
            group = summary[summary.method == method_name].sort_values("missing_rate")
            ax.errorbar(group.missing_rate, group[f"{metric}_mean"], yerr=group[f"{metric}_sem"],
                        marker="o", capsize=3, label=method_name, color=METHOD_COLORS.get(method_name))
        ax.set_title(dataset_name)
        ax.set_xlabel("Missing rate")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def _dataset_title(df: pd.DataFrame, metric: str, title: str = None) -> str:
    if title:
        return title
    dataset_names = ", ".join(df["dataset"].unique())
    return f"{dataset_names} — {metric}"


def plot_metric_vs_missing_rate(df: pd.DataFrame, metric: str = "accuracy", output_path: str = None, title: str = None):
    import matplotlib.pyplot as plt

    summary = summarize_results(df, metrics=(metric,))
    fig, ax = plt.subplots()
    for method_name, group in summary.groupby("method"):
        group = group.sort_values("missing_rate")
        ax.errorbar(group.missing_rate, group[f"{metric}_mean"], yerr=group[f"{metric}_sem"],
                    marker="o", capsize=3, label=method_name, color=METHOD_COLORS.get(method_name))
    ax.set_xlabel("Missing rate")
    ax.set_ylabel(metric)
    ax.set_title(_dataset_title(df, metric, title))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_metric_bar_chart(df: pd.DataFrame, metric: str = "accuracy", output_path: str = None, title: str = None):
    import matplotlib.pyplot as plt

    summary = summarize_results(df, metrics=(metric,))
    method_names = summary["method"].unique()
    missing_rates = sorted(summary["missing_rate"].unique())
    x = np.arange(len(missing_rates))
    width = 0.8 / len(method_names)

    fig, ax = plt.subplots()
    for i, method_name in enumerate(method_names):
        sub = (summary[summary.method == method_name]
               .set_index("missing_rate").reindex(missing_rates))
        ax.bar(x + i * width, sub[f"{metric}_mean"], width,
               yerr=sub[f"{metric}_sem"], capsize=3, label=method_name,
               color=METHOD_COLORS.get(method_name))

    ax.set_xticks(x + width * (len(method_names) - 1) / 2)
    ax.set_xticklabels([str(r) for r in missing_rates])
    ax.set_xlabel("Missing rate")
    ax.set_ylabel(metric)
    ax.set_title(_dataset_title(df, metric, title))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def to_latex_table(summary_df: pd.DataFrame, output_path: str = None) -> str:
    latex = summary_df.round(4).to_latex(index=False)
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
    return latex
