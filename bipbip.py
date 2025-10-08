#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random series plotter (seaborn) with optional state overlay.

- Loads a 2D NumPy array of shape (n_series, n_steps) or (n_steps, n_series).
- Randomly selects one series, plots the first STEPS_TO_PLOT points (e.g., 5,000).
- If SHOW_STATE is True:
    * Loads params dict from PARAMS_PATH and reads params["Z"] (torch.tensor)
      with the same shape as the original array: (n_series, n_steps).
    * Uses the same randomly selected series index to fetch its states for the
      first STEPS_TO_PLOT points.
    * Shades the background by state segments and draws a “state strip” below.
- Saves the figure as <PLOT_NAME>.png to SAVE_DIR.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== USER SETTINGS =====================
PLOT_NAME = "random_series_plot"   # Title and filename stem (e.g., random_series_plot.png)
DATA_PATH = "series.npy"           # Path to .npy array of series
SAVE_DIR  = "."                    # Directory to save the image
STEPS_TO_PLOT = 5000               # Number of steps to plot (from the start)
SEED = None                        # Reproducibility; set to an int or keep None

# ---- State overlay options ----
SHOW_STATE  = False                # Set True to overlay latent state
PARAMS_PATH = "params.pt"          # Path to a torch file with dict containing key "Z"
Z_KEY       = "Z"                  # Key in params dict that holds the torch.tensor
# =========================================================

def _ensure_series_time_layout(arr):
    """
    Ensure array is (n_series, n_steps).
    If time appears on axis 0 (e.g., shape (n_steps, n_series)), transpose.
    Heuristic: we assume n_steps >= n_series in typical use.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    if arr.shape[1] >= arr.shape[0]:
        return arr, False  # (n_series, n_steps), not transposed
    else:
        return arr.T, True  # transposed

def _load_states_like_data(params_path, z_key, transposed_like_data):
    """
    Load params dict and extract Z torch.tensor, converting to numpy.
    If data was transposed, transpose Z too to keep (n_series, n_steps).
    """
    import torch  # imported only if needed
    params = torch.load(params_path, map_location="cpu")
    if z_key not in params:
        raise KeyError(f'Key "{z_key}" not found in params dict.')
    Z = params[z_key]
    if not hasattr(Z, "shape") or Z.ndim != 2:
        raise ValueError(f'params["{z_key}"] must be a 2D tensor; got shape {getattr(Z, "shape", None)}')
    Z_np = Z.detach().cpu().numpy()
    if transposed_like_data:
        Z_np = Z_np.T
    return Z_np

def _run_length_encode(arr1d):
    """
    Return (starts, ends, values) for contiguous runs in a 1D array.
    Each run covers indices [start, end) (end exclusive).
    """
    if arr1d.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    change_points = np.where(np.diff(arr1d) != 0)[0] + 1
    starts = np.r_[0, change_points]
    ends   = np.r_[change_points, arr1d.size]
    vals   = arr1d[starts]
    return starts, ends, vals

def main():
    # Load data
    arr = np.load(DATA_PATH)
    series_by_time, transposed = _ensure_series_time_layout(arr)
    n_series, n_steps = series_by_time.shape

    if n_steps < STEPS_TO_PLOT:
        raise ValueError(f"Time dimension has only {n_steps} steps; need at least {STEPS_TO_PLOT}.")

    # Pick a random series
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_series)
    y = series_by_time[idx, :STEPS_TO_PLOT]
    x = np.arange(STEPS_TO_PLOT)

    # Prepare seaborn
    sns.set(style="whitegrid")

    if SHOW_STATE:
        # Load state tensor "Z" matching data shape/orientation
        Z_np = _load_states_like_data(PARAMS_PATH, Z_KEY, transposed)
        if Z_np.shape != arr.shape and not transposed:
            # If data wasn't transposed and shapes differ, check if we still need to transpose Z
            if Z_np.T.shape == series_by_time.shape:
                Z_np = Z_np.T
            else:
                raise ValueError(
                    f'Shape mismatch: data {series_by_time.shape} vs params["{Z_KEY}"] {Z_np.shape}.'
                )
        # Now Z_np should be (n_series, n_steps)
        if Z_np.shape[1] < STEPS_TO_PLOT:
            raise ValueError(
                f'params["{Z_KEY}"] time dimension has only {Z_np.shape[1]} steps; need at least {STEPS_TO_PLOT}.'
            )
        states = Z_np[idx, :STEPS_TO_PLOT]

        # Build palette for unique states encountered in this window
        unique_states = np.unique(states)
        k = unique_states.size
        palette = sns.color_palette("tab10", n_colors=max(k, 1))
        state_to_color = {s: palette[i % len(palette)] for i, s in enumerate(unique_states)}
        state_to_ord   = {s: i for i, s in enumerate(unique_states)}  # for the strip

        # Figure with two rows: line + state strip
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        # 1) Plot the series
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_ylabel("Value")
        ax.set_title(PLOT_NAME + f"  (series index = {idx})")

        # 2) Shade background by contiguous state segments
        starts, ends, vals = _run_length_encode(states)
        for s, e, v in zip(starts, ends, vals):
            ax.axvspan(s, e, color=state_to_color[v], alpha=0.15, linewidth=0)

        # Make a legend for states
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=state_to_color[s], label=f"State {s}") for s in unique_states]
        ax.legend(handles=handles, title="States", ncols=min(4, len(handles)), loc="upper left")

        # 3) Draw a compact “state strip” below using seaborn heatmap
        # Map actual states to ordinal 0..k-1 so colors are consistent with palette
        states_ord = np.vectorize(state_to_ord.get)(states)
        # To use palette in heatmap, build a ListedColormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(palette[:k])

        # seaborn.heatmap expects 2D; we provide shape (1, STEPS_TO_PLOT)
        sns.heatmap(
            states_ord[np.newaxis, :],
            ax=ax2,
            cmap=cmap,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            linewidths=0,
            linecolor=None,
        )
        ax2.set_ylabel("State", rotation=0, labelpad=30)
        ax2.set_yticks([])  # keep it clean
        ax.set_xlim(0, STEPS_TO_PLOT - 1)
        ax.set_xlabel("Step")

        plt.tight_layout()

    else:
        # Simple single plot without states
        plt.figure(figsize=(12, 4))
        sns.lineplot(x=x, y=y)
        plt.title(PLOT_NAME + f"  (series index = {idx})")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.tight_layout()

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f"{PLOT_NAME}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot for series index {idx} to: {out_path}")

if __name__ == "__main__":
    main()
