import pandas as pd
import numpy as np

def calculate_hawkes_approximation(
    matrix_df: pd.DataFrame, 
    bid_cols: list, 
    ask_cols: list, 
    distance_decay: float = 1.5, 
    decay_span: int = 10
) -> pd.DataFrame:
    """
    Calculates a discrete approximation of Hawkes processes for Level 2 (L2) snapshots.
    
    Parameters:
    - matrix_df: DataFrame containing LOB snapshots (must be chronologically sorted).
    - bid_cols: Sorted list of column names representing Bid depths.
    - ask_cols: Sorted list of column names representing Ask depths.
    - distance_decay: Penalty parameter for the distance from the mid-price (spatial kernel).
    - decay_span: Time window for the decay function (exponential temporal kernel).
    
    Returns:
    - pd.DataFrame: The original DataFrame enriched with 'hawkes_bid', 'hawkes_ask', 
                    and 'hawkes_imbalance' features.
    """
    
    # A. Volume differences (identifying the arrival of new limit orders)
    # We clip at 0 to only capture order additions (inflows), ignoring cancellations/executions.
    bids_diff = matrix_df[bid_cols].diff().clip(lower=0).fillna(0)
    asks_diff = matrix_df[ask_cols].diff().clip(lower=0).fillna(0)
    
    hawkes_intensity_bid = np.zeros(len(matrix_df))
    hawkes_intensity_ask = np.zeros(len(matrix_df))
    
    K = min(len(bid_cols), len(ask_cols))

    for k in range(K):
        dist_k = float(bid_cols[k].split('_')[2])
        weight_k = np.exp(-distance_decay * dist_k)
        
        hawkes_intensity_bid += bids_diff[bid_cols[k]] * weight_k
        hawkes_intensity_ask += asks_diff[ask_cols[k]] * weight_k

    # Temporal Decay using EMA
    hawkes_bids_ema = pd.Series(hawkes_intensity_bid, index=matrix_df.index).ewm(span=decay_span, adjust=False).mean()
    hawkes_asks_ema = pd.Series(hawkes_intensity_ask, index=matrix_df.index).ewm(span=decay_span, adjust=False).mean()
    
    matrix_df['hawkes_bid'] = hawkes_bids_ema
    matrix_df['hawkes_ask'] = hawkes_asks_ema
    matrix_df['hawkes_imbalance'] = (matrix_df['hawkes_bid'] - matrix_df['hawkes_ask']) / \
                                    (matrix_df['hawkes_bid'] + matrix_df['hawkes_ask'] + 1e-8)
                                    
    return matrix_df