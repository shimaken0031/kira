import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure

# ==========================================
# 0. パラメータ設定（バランス最適化版）
# ==========================================
PIXEL_TO_MICRON = 0.07

# 判定条件（短め・V字・淡い菌体もバランス良く拾う）
MIN_LENGTH_UM = 0.6     # 0.6μm以上の菌体を救出
MAX_LENGTH_UM = 12.0    # 12.0μm超の大きなデブリを除外
MIN_AREA_PX = 15        # 小さめの菌体も拾えるよう設定
MAX_AREA_PX = 800       # 巨大デブリを除外
MIN_ASPECT_RATIO = 1.4  # 曲がり菌・V字菌・短め菌を救出
MIN_SOLIDITY = 0.75     # 充填率を緩和して変形菌体を救出
CLEAR_BORDER = True     # 画面端の半欠け菌を除外


# ==========================================
# 1. 単一画像の解析関数（Warning修正済み）
# ==========================================
def analyze_single_image(file_path):
    try:
        with Image.open(file_path) as img:
            img_np = np.array(img.convert("L"))
    except Exception as e:
        print(f"  [エラー] 画像を開けませんでした: {e}")
        return None, None, None

    # 前処理：コントラスト自動補正
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 背景むらキャンセル処理
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    top_hat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_bg)

    # Otsu二値化
    _, thresh = cv2.threshold(
        top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 微小ノイズのみ最小限で削除
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    if CLEAR_BORDER:
        h, w = thresh.shape
        cv2.rectangle(thresh, (0, 0), (w - 1, h - 1), 0, 2)

    # 領域抽出 & 判定
    labels = measure.label(thresh, background=0)
    props = measure.regionprops(labels)

    valid_cells = []
    mask_display = np.zeros_like(labels)

    for p in props:
        # scikit-image 最新バージョン用 (FutureWarning回避)
        major_len = (
            getattr(p, "axis_major_length", None)
            if hasattr(p, "axis_major_length")
            else p.major_axis_length
        )
        minor_len = (
            getattr(p, "axis_minor_length", None)
            if hasattr(p, "axis_minor_length")
            else p.minor_axis_length
        )

        length_um = major_len * PIXEL_TO_MICRON
        width_um = minor_len * PIXEL_TO_MICRON
        area_um2 = p.area * (PIXEL_TO_MICRON**2)
        aspect_ratio = major_len / minor_len if minor_len > 0 else 0
        solidity = p.solidity

        is_length_ok = MIN_LENGTH_UM <= length_um <= MAX_LENGTH_UM
        is_area_ok = MIN_AREA_PX <= p.area <= MAX_AREA_PX
        is_elongated = aspect_ratio >= MIN_ASPECT_RATIO
        is_solid = solidity >= MIN_SOLIDITY

        if is_length_ok and is_area_ok and is_elongated and is_solid:
            valid_cells.append(
                {
                    "File_Name": os.path.basename(file_path),
                    "ID": p.label,
                    "Length_um": round(length_um, 2),
                    "Width_um": round(width_um, 2),
                    "Aspect_Ratio": round(aspect_ratio, 2),
                    "Area_um2": round(area_um2, 2),
                }
            )
            mask_display[labels == p.label] = p.label

    df_single = pd.DataFrame(valid_cells)
    return df_single, img_np, mask_display


# ==========================================
# 2. 分布グラフの生成・保存関数
# ==========================================
def generate_distribution_plots(df_total):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        df_total["Length_um"],
        bins=25,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    mean_len = df_total["Length_um"].mean()
    axes[0].axvline(
        mean_len,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_len:.2f} μm",
    )
    axes[0].set_title("Bifidobacterium Length Distribution", fontsize=14)
    axes[0].set_xlabel("Length (μm)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].legend(fontsize=12)

    axes[1].hist(
        df_total["Width_um"],
        bins=20,
        color="salmon",
        edgecolor="black",
        alpha=0.7,
    )
    mean_wid = df_total["Width_um"].mean()
    axes[1].axvline(
        mean_wid,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_wid:.2f} μm",
    )
    axes[1].set_title("Bifidobacterium Width Distribution", fontsize=14)
    axes[1].set_xlabel("Width (μm)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    graph_filename = "bifido_size_distribution.png"
    plt.savefig(graph_filename, dpi=300)
    print(f"\n ※ サイズ分布グラフを '{graph_filename}' に保存しました。")
    plt.show()


# ==========================================
# 3. メイン処理
# ==========================================
def main():
    processed_files = set()
    file_summary_list = []
    all_accumulated_cells = []

    print("==========================================")
    print("   ビフィズス菌 複数画像解析システム (Warning修正完了版)")
    print("==========================================")

    root = tk.Tk()
    root.withdraw()

    while True:
        file_paths = filedialog.askopenfilenames(
            title="解析する画像を選択してください（キャンセルで終了）",
            filetypes=[
                ("画像ファイル", "*.bmp *.tmp *.png *.jpg *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )

        if not file_paths:
            print("\n解析処理を終了します。")
            break

        print(f"\n--- 今回選択された枚数: {len(file_paths)} 枚 ---")

        for file_path in file_paths:
            file_name = os.path.basename(file_path)

            if file_name in processed_files:
                print(f"[スキップ] すでに解析済みです: {file_name}")
                continue

            start_time = time.time()
            print(f"\n[解析開始] {file_name}")

            df_single, img_np, mask_display = analyze_single_image(file_path)

            if df_single is None:
                continue

            processed_files.add(file_name)
            elapsed_time = time.time() - start_time
            count = len(df_single)

            if count > 0:
                avg_len = df_single["Length_um"].mean()
                avg_wid = df_single["Width_um"].mean()
                all_accumulated_cells.append(df_single)
            else:
                avg_len = 0.0
                avg_wid = 0.0

            file_summary_list.append(
                {
                    "File_Name": file_name,
                    "Count": count,
                    "Avg_Length_um": round(avg_len, 2),
                    "Avg_Width_um": round(avg_wid, 2),
                }
            )

            print(f" └ 処理時間     : {elapsed_time:.3f} 秒")
            print(f" └ 検出された個体数: {count} 個")

            fig, ax = plt.subplots(1, 2, figsize=(11, 5))
            ax[0].imshow(img_np, cmap="gray")
            ax[0].set_title(f"Original: {file_name}")
            ax[0].axis("off")

            ax[1].imshow(mask_display, cmap="jet")
            ax[1].set_title(f"Detected: {count} cells ({elapsed_time:.2f}s)")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()

    if not file_summary_list:
        return

    print("\n" + "=" * 65)
    print("      【 1. 解析ファイル別 結果一覧 】")
    print("=" * 65)
    df_file_summary = pd.DataFrame(file_summary_list)
    print(
        df_file_summary.to_string(
            index=False,
            header=["ファイル名", "検出個数", "平均長さ(μm)", "平均幅(μm)"],
        )
    )

    print("\n" + "=" * 65)
    print("      【 2. 全データの総合平均・統計 】")
    print("=" * 65)
    total_files = len(file_summary_list)
    total_cells = df_file_summary["Count"].sum()
    print(f" 処理済み画像枚数      : {total_files} 枚")
    print(f" 検出菌体の総数        : {total_cells} 個")
    print(f" 画像1枚あたりの平均個数: {total_cells / total_files:.2f} 個")

    if all_accumulated_cells:
        df_total = pd.concat(all_accumulated_cells, ignore_index=True)
        print("-" * 65)
        print(f" 【全菌体総合平均】 長さ: {df_total['Length_um'].mean():.2f} μm")
        print(f" 【全菌体総合平均】 幅  : {df_total['Width_um'].mean():.2f} μm")
        print("-" * 65)

        output_csv = "bifido_summary_result.csv"
        df_total.to_csv(output_csv, index=False, encoding="utf-8-sig")
        generate_distribution_plots(df_total)


if __name__ == "__main__":
    main()