"""Standardized visualization for hazard models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_data import PROCESS_COLORS, PROCESS_ORDER
from utils_metrics import compute_warning_thresholds, assign_warning_level

_WARN_COLORS = {"Verde": "#2ecc71", "Giallo": "#f1c40f", "Arancione": "#e67e22", "Rosso": "#e74c3c"}
_WARN_LEVELS = ["Verde", "Giallo", "Arancione", "Rosso"]

def _handle_save(fig: plt.Figure, save_path: str | None = None):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

def plot_roc_curves(perf_dict: dict, title: str = "Fitting performance (ROC)", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    for pname in PROCESS_ORDER:
        if pname not in perf_dict: continue
        r = perf_dict[pname]
        ax.plot(r["fpr"], r["tpr"], color=PROCESS_COLORS[pname], lw=1.5,
                label=f"{pname} (AUC = {r['AUROC']:.2f})")
        # Find best point
        best_thr = r["threshold"]
        idx = np.argmin(np.abs(np.linspace(0.01, 0.99, 99) - best_thr)) # Approximate
        # Real best point from metrics
        # For display, just scatter the point where F2 was maximized
        # Note: the perf_dict should have y_prob and y_true if we want precise best point
    
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)",
           title=title, xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(loc="lower right", framealpha=0.7)
    _handle_save(fig, save_path)

def plot_pr_curves(perf_dict: dict, title: str = "Precision-Recall performance", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for pname in PROCESS_ORDER:
        if pname not in perf_dict: continue
        r = perf_dict[pname]
        ax.plot(r["pr_rec"], r["pr_prec"], color=PROCESS_COLORS[pname], lw=1.5,
                label=f"{pname} (AUPRC = {r['AUPRC']:.3f})")
        ax.axhline(r["prevalence"], color=PROCESS_COLORS[pname], ls="--", alpha=0.6, lw=1)
    
    ax.set(xlabel="Recall (Sensitivity)", ylabel="Precision",
           title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="center right", framealpha=0.8)
    fig.text(0.5, 0.01, "Dashed lines: no-skill baseline (prevalence)", ha="center", fontsize=9, color="grey")
    _handle_save(fig, save_path)

def plot_reliability_diagrams(perf_dict: dict, title: str = "Reliability diagram (Calibration)", save_path: str | None = None):
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, pname in zip(axes, PROCESS_ORDER):
        if pname not in perf_dict: 
            ax.set_visible(False)
            continue
        r = perf_dict[pname]
        # Recalculate curve for plot
        y_true = r.get("y_true")
        y_prob = r.get("y_prob")
        if y_true is None or y_prob is None:
            ax.text(0.5, 0.5, "Missing y_true/y_prob", ha="center")
            continue
            
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="Perfect")
        ax.plot(prob_pred, prob_true, "o-", color=PROCESS_COLORS[pname], lw=2, ms=5)
        
        ax.fill_between(prob_pred, prob_true, prob_pred, where=(prob_true < prob_pred), alpha=0.18, color="#e74c3c")
        ax.fill_between(prob_pred, prob_true, prob_pred, where=(prob_true >= prob_pred), alpha=0.18, color="#2ecc71")
        
        ax.axhline(r["prevalence"], ls=":", color="grey", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", aspect="equal", title=pname)
    
    axes[0].set_ylabel("Fraction of positives")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _handle_save(fig, save_path)

def plot_importance(model_dict: dict, title: str = "Variable Importance", save_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for ax, pname in zip(axes, PROCESS_ORDER):
        if pname not in model_dict:
            ax.set_visible(False)
            continue
        model = model_dict[pname]
        g = model.explain_global()
        gdata = g.data()
        df = pd.DataFrame({"Variable": gdata["names"], "Importance": gdata["scores"]}).sort_values("Importance")
        
        ax.barh(df["Variable"], df["Importance"], color=PROCESS_COLORS[pname], alpha=0.8)
        ax.set_title(pname, fontweight="bold", color=PROCESS_COLORS[pname])
        ax.set_xlabel("Mean absolute score")
    
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _handle_save(fig, save_path)

def plot_partial_effects(model, titles: dict, pname: str, save_path: str | None = None):
    g = model.explain_global()
    gdata = g.data()
    feat_names = list(gdata["names"])
    n_cols = 4
    n_rows = int(np.ceil(len(feat_names) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    color = PROCESS_COLORS[pname]
    
    for i, fname in enumerate(feat_names):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        
        # Extract effect data safely
        data = g.data(i)
        if "data" in data and isinstance(data["data"], dict): data = data["data"]
        
        names = data.get("names", [])
        scores_raw = data.get("scores", [])
        scores = np.asarray(scores_raw, dtype=float)

        # Interaction terms produce 2D score matrices — render as heatmap.
        if scores.ndim == 2:
            ax.imshow(scores.T, aspect="auto", cmap="RdBu_r", origin="lower",
                      interpolation="nearest")
            ax.set_xlabel(titles.get(fname, fname), fontsize=7)
            ax.set_title(f"{chr(97 + i)})", loc="left", fontweight="bold", fontsize=10)
            continue

        is_cat = len(names) > 0 and isinstance(names[0], str)

        if is_cat:
            ax.barh(np.arange(len(names)), scores, color=color, alpha=0.7)
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels(names, fontsize=7)
        else:
            x_vals = np.asarray(names, dtype=float)
            if len(x_vals) == len(scores) + 1:
                x_vals = (x_vals[:-1] + x_vals[1:]) / 2.0
            ax.plot(x_vals, scores, color=color, lw=1.5)
            if "upper_bounds" in data and "lower_bounds" in data:
                ax.fill_between(x_vals, data["lower_bounds"], data["upper_bounds"],
                                color=color, alpha=0.15)
            ax.axhline(0, color="grey", ls="--", lw=0.5)

        ax.set_xlabel(titles.get(fname, fname), fontsize=8)
        ax.set_title(f"{chr(97 + i)})", loc="left", fontweight="bold", fontsize=10)

    for j in range(len(feat_names), n_rows * n_cols):
        axes.flatten()[j].set_visible(False)
        
    fig.suptitle(f"Partial Effects - {pname}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _handle_save(fig, save_path)

def plot_warning_levels(perf_dict: dict, save_path: str | None = None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, pname in enumerate(PROCESS_ORDER):
        if pname not in perf_dict: continue
        r = perf_dict[pname]
        y_true = r["y_true"]
        y_prob = r["y_prob"]
        t1, t2, t3 = compute_warning_thresholds(r["prevalence"])
        
        # Density plot with warning-level colour bands
        _boundaries = [0, t1, t2, t3, 1.0]

        ax_top = axes[0, col]
        # Draw background bands first so histograms render on top
        for lo, hi, wl in zip(_boundaries[:-1], _boundaries[1:], _WARN_LEVELS):
            ax_top.axvspan(lo, hi, alpha=0.15, color=_WARN_COLORS[wl], zorder=0)
        # Threshold boundary lines
        for thr, wl in zip([t1, t2, t3], ["Giallo", "Arancione", "Rosso"]):
            ax_top.axvline(thr, color=_WARN_COLORS[wl], ls="--", lw=1.2, zorder=1)

        ax_top.hist(y_prob[y_true==0], bins=50, density=True, alpha=0.55, color="steelblue", label="No event", zorder=2)
        ax_top.hist(y_prob[y_true==1], bins=50, density=True, alpha=0.55, color="coral", label="Event", zorder=2)
        ax_top.set_xlim(0, 1)
        ax_top.set_xlabel("Predicted probability")
        ax_top.set_title(f"{pname}\nπ={r['prevalence']:.4f}  |  t₁={t1:.3f}  t₂={t2:.3f}  t₃={t3:.3f}")
        ax_top.legend(fontsize=8)
        
        # Hit rates
        ax_bot = axes[1, col]
        rates = []
        for lvl in _WARN_LEVELS:
            mask = np.array([assign_warning_level(p, t1, t2, t3) == lvl for p in y_prob])
            rate = float(y_true[mask].mean()) if mask.any() else 0.0
            rates.append(rate)
        ax_bot.bar(_WARN_LEVELS, rates,
                   color=[_WARN_COLORS[wl] for wl in _WARN_LEVELS], edgecolor="black", alpha=0.85)
        ax_bot.axhline(r["prevalence"], ls="--", color="grey", lw=1,
                       label=f"Prevalence={r['prevalence']:.4f}")
        ax_bot.set_xlabel("Warning level")
        ax_bot.set_ylabel("Event rate")
        ax_bot.set_ylim(0, 1)
        ax_bot.legend(fontsize=7)

    fig.suptitle("Alarm Level — Province of Bolzano (Allegato D)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _handle_save(fig, save_path)
