"""
Bonferroni correction cho các so sánh paired t-test đã có trong results/*.csv.
KHÔNG chạy lại thí nghiệm — chỉ đọc dữ liệu đã có sẵn.

Hai họ so sánh (family) được hiệu chỉnh RIÊNG, mỗi họ có m riêng:
  1. KNN-Direct (Euclidean) vs KNN-Direct-Manhattan, trên cả 5 dataset x 5 missing rate.
  2. KNN-IC vs KNN-Direct, chỉ trên Parkinson x 5 missing rate (Thảo luận ý 4).
"""
import os
import pandas as pd

from experiment_utils import load_results, compare_methods_across_datasets

RESULTS_DIR = "results"
OUTPUT_DIR = "results"

names = ["Toxicity", "Darwin", "Micromass", "Parkinson", "Parkinson2clc"]


def load_all_datasets():
    dfs = {}
    for name in names:
        path = os.path.join(RESULTS_DIR, f"{name.lower()}_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy {path} — kiểm tra lại tên file trong {RESULTS_DIR}/")
        dfs[name] = load_results(path)
    return dfs


def bonferroni_correct(tbl: pd.DataFrame) -> pd.DataFrame:
    m = len(tbl)
    alpha_corrected = 0.05 / m
    tbl = tbl.copy()
    tbl["p_bonf"] = (tbl["ttest_p"] * m).clip(upper=1.0)
    tbl["sig_raw"] = tbl["ttest_p"] < 0.05
    tbl["sig_bonf"] = tbl["p_bonf"] < 0.05
    return tbl, m, alpha_corrected


def report(title: str, tbl: pd.DataFrame, out_path: str):
    tbl, m, alpha_corrected = bonferroni_correct(tbl)

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"m (số so sánh) = {m}  |  alpha hiệu chỉnh = 0.05/{m} = {alpha_corrected:.6f}")

    print("\nBảng đầy đủ:")
    cols = ["dataset", "missing_rate", "ttest_p", "p_bonf", "sig_raw", "sig_bonf"]
    print(tbl[cols].to_string(index=False))

    print("\nTổng hợp theo dataset (số mức có ý nghĩa, trước/sau hiệu chỉnh):")
    summary = tbl.groupby("dataset")[["sig_raw", "sig_bonf"]].sum()
    print(summary.to_string())

    tbl[cols].to_csv(out_path, index=False)
    print(f"\n-> đã ghi bảng đầy đủ ra {out_path}")
    return tbl, summary


if __name__ == "__main__":
    dfs = load_all_datasets()

    # 1. KNN-Direct (Euclidean) vs KNN-Direct-Manhattan, tất cả dataset
    tbl_manhattan = compare_methods_across_datasets(dfs, "KNN-Direct", "KNN-Direct-Manhattan")
    tbl_manhattan, summary_manhattan = report(
        "1) KNN-Direct (Euclidean) vs KNN-Direct-Manhattan — tất cả 5 dataset",
        tbl_manhattan,
        os.path.join(OUTPUT_DIR, "bonferroni_manhattan_vs_euclidean.csv"),
    )

    # 2. KNN-IC vs KNN-Direct, chỉ Parkinson (Thảo luận ý 4)
    tbl_parkinson = compare_methods_across_datasets({"Parkinson": dfs["Parkinson"]}, "KNN-IC", "KNN-Direct")
    tbl_parkinson, summary_parkinson = report(
        "2) KNN-IC vs KNN-Direct (Euclidean) — chỉ Parkinson (Thảo luận ý 4)",
        tbl_parkinson,
        os.path.join(OUTPUT_DIR, "bonferroni_knnic_vs_direct_parkinson.csv"),
    )

    print(f"\n{'=' * 70}")
    print("Lưu ý: results/*.csv hiện tại là dữ liệu TRƯỚC khi sửa 2 lỗi pipeline gần nhất")
    print("(patient-level leakage cho Parkinson/Parkinson2clc, và thứ tự split->missing->scale).")
    print("Cần chạy lại thí nghiệm để có số liệu Bonferroni cuối cùng dùng cho bài báo.")
    print(f"{'=' * 70}")
