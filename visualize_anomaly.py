import matplotlib.pyplot as plt
import numpy as np

def visualize_anomaly(matrix_df, target_timestamp):
  print(f"Generating Order Book Heatmap for {target_timestamp}...")
  
  # Extract just the row for our target anomaly
  anomaly_row = matrix_df.loc[target_timestamp]
  
  # Separate the bid and ask column names
  bid_cols = [col for col in matrix_df.columns if 'bid_depth' in col]
  ask_cols = [col for col in matrix_df.columns if 'ask_depth' in col]
  
  # Sort them so they plot correctly from the mid-price outward
  # Bids: 0.2, 1.0, 2.0, 3.0...
  bid_cols_sorted = sorted(bid_cols, key=lambda x: float(x.split('_')[2]))
  ask_cols_sorted = sorted(ask_cols, key=lambda x: float(x.split('_')[2]))
  
  # Extract the actual volume values
  bid_vols = anomaly_row[bid_cols_sorted].values
  ask_vols = anomaly_row[ask_cols_sorted].values
  
  # X-axis labels (Percentage from mid-price)
  percentages = [x.split('_')[2] + '%' for x in bid_cols_sorted]
  
  # Create the visualization
  fig, ax = plt.subplots(figsize=(12, 6))
  
  # Plot Bids (Green, pointing left/down) and Asks (Red, pointing right/up)
  x_pos = np.arange(len(percentages))
  width = 0.4
  
  ax.bar(x_pos - width/2, bid_vols, width, label='Bids (Buyers)', color='green', alpha=0.7)
  ax.bar(x_pos + width/2, ask_vols, width, label='Asks (Sellers)', color='red', alpha=0.7)
  
  ax.set_ylabel('Total Volume Resting')
  ax.set_xlabel('Distance from Mid-Price')
  ax.set_title(f'Limit Order Book Structural Anomaly at {target_timestamp}\n(Notice the massive volume disparity)')
  ax.set_xticks(x_pos)
  ax.set_xticklabels(percentages)
  ax.legend()
  ax.grid(True, linestyle='--', alpha=0.5)
  
  plt.tight_layout()
  plt.show()