import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    required = {"configuration", "Time", "CF"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Coerce numeric where possible (non-numeric CF rows become NaN)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["CF"] = pd.to_numeric(df["CF"], errors="coerce")

    # Drop rows with missing values for each plot separately
    df_time = df.dropna(subset=["configuration", "Time"]).copy()
    df_cf = df.dropna(subset=["configuration", "CF"]).copy()

    # Keep configuration order as it appears in the CSV
    config_order = list(dict.fromkeys(df["configuration"].astype(str).tolist()))

    # ---- Boxplot: CF ----
    data_cf = [df_cf.loc[df_cf["configuration"] == c, "CF"].values for c in config_order]
    plt.figure()
    bp = plt.boxplot(
        data_cf,
        tick_labels=config_order,
        showfliers=True,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=6
        )
    )

    # Add numeric labels for the mean
    for i, values in enumerate(data_cf, start=1):
        if len(values) == 0:
            continue
        mean_val = np.mean(values)

        plt.text(
            i,                      # x position (box index)
            mean_val,               # y position (mean)
            f"{mean_val:.3g}",      # formatted value
            ha="left",
            va="bottom",
            fontsize=9,
            color="red"
        )

    plt.xlabel("Configuration")
    plt.ylabel("CF")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_png1 = "./plots/CF_combined_py.png"
    plt.savefig(out_png1, dpi=150)

    # ---- Boxplot: Time ----
    data_time = [df_time.loc[df_time["configuration"] == c, "Time"].values for c in config_order]
    plt.figure()
    bp = plt.boxplot(
        data_time,
        tick_labels=config_order,
        showfliers=True,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=6
        )
    )

    for i, values in enumerate(data_time, start=1):
        if len(values) == 0:
            continue
        mean_val = np.mean(values)

        plt.text(
            i,
            mean_val,
            f"{mean_val:.3g}ms",
            ha="left",
            va="bottom",
            fontsize=9,
            color="red"
        )

    plt.xlabel("Configuration")
    plt.ylabel("Time (milliseconds)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_png2 = "./plots/Time_combined_py.png"
    plt.savefig(out_png2, dpi=150)

if __name__ == "__main__":
    main("./csv/results_py.csv")
