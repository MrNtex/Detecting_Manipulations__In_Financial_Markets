import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_orderbook_heatmap(feature_file, anomaly_time, window_minutes=5):

    df = pd.read_parquet(feature_file)

    df.index = pd.to_datetime(df.index)

    anomaly_time = pd.to_datetime(anomaly_time)

    window = df.loc[
        anomaly_time - pd.Timedelta(minutes=window_minutes):
        anomaly_time + pd.Timedelta(minutes=window_minutes)
    ]

    # Select only depth columns
    depth_cols = [c for c in window.columns if "depth" in c]

    depth_matrix = window[depth_cols]

    # Convert to numpy matrix for heatmap
    data = depth_matrix.T.values

    plt.figure(figsize=(14,6))

    plt.imshow(
        data,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="Liquidity Depth")

    plt.yticks(
        range(len(depth_cols)),
        depth_cols
    )

    # mark anomaly moment
    timestamps = window.index
    anomaly_idx = np.argmin(np.abs(timestamps - anomaly_time))

    plt.axvline(anomaly_idx, linestyle="--", label="Anomaly")

    plt.title("Order Book Liquidity Heatmap Around Anomaly")

    plt.xlabel("Time")
    plt.ylabel("Order Book Levels")

    plt.legend()

    plt.tight_layout()

    plt.show()


def visualize_top_heatmaps(anomaly_file):

    anomaly_file = Path(anomaly_file)

    date = anomaly_file.stem.split("_")[1]

    feature_file = Path(f"data/features/features_{date}.parquet")

    anomalies = pd.read_parquet(anomaly_file)

    anomalies = anomalies.sort_values("anomaly_score").head(3)

    for ts in anomalies.index:

        print("Visualizing anomaly:", ts)

        visualize_orderbook_heatmap(
            feature_file,
            ts
        )


if __name__ == "__main__":

    visualize_top_heatmaps(
        "data/anomalies/anomalies_2023-10-16.parquet"
    )