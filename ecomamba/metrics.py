"""Evaluation metrics for EF regression.

Beyond the three metrics the assignment lists (MAE / RMSE / R^2) we report several
clinically motivated ones; each is justified in ``docs/DESIGN.md``:

* **Pearson r** and **Spearman rho** - correlation is scale free; Spearman additionally
  ignores monotone rescaling, which matters because EF predictions are systematically
  compressed toward the mean when the training set is small.
* **Bland-Altman** bias, SD of differences and 95% limits of agreement - the standard
  agreement analysis in cardiac imaging; a model can have a low MAE yet a
  non-zero systematic bias, which is what a clinician would see.
* **Clinical agreement rates** - % of clips within +-5 and +-10 EF points.  These are the
  thresholds used in the echo literature and are far more interpretable than MAE.
* **Subgroup MAE** for reduced EF (EF < 40) - heart-failure patients are the clinically
  relevant population and the hardest regime; a model that only fits normal EFs can
  still look good on the global MAE.
* **Regression calibration** (slope / intercept of true on predicted) - detects the
  classic "regression to the mean" failure of small-data regressors.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    return _pearson(rank(a), rank(b))


def compute_metrics(y_true, y_pred, ef_min: float = 0.0, ef_max: float = 100.0) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert y_true.shape == y_pred.shape

    err = y_pred - y_true
    ae = np.abs(err)
    m = {
        "n": int(y_true.size),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": _r2(y_true, y_pred),
        "pearson": _pearson(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "bias": float(err.mean()),
        "sd_diff": float(err.std(ddof=1)) if err.size > 1 else float("nan"),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
    }
    m["loa_lower"] = m["bias"] - 1.96 * m["sd_diff"]
    m["loa_upper"] = m["bias"] + 1.96 * m["sd_diff"]
    m["agree_5"] = float((ae <= 5.0).mean())
    m["agree_10"] = float((ae <= 10.0).mean())

    # regression calibration: y_true ~ a + b * y_pred
    if y_pred.std() > 1e-9:
        b, a = np.polyfit(y_pred, y_true, 1)
        m["calib_slope"] = float(b)
        m["calib_intercept"] = float(a)
    else:
        m["calib_slope"] = m["calib_intercept"] = float("nan")

    # clinically important subgroup: reduced ejection fraction
    mask = y_true < 40.0
    m["n_low_ef"] = int(mask.sum())
    m["mae_low_ef"] = float(ae[mask].mean()) if mask.any() else float("nan")
    mask_hi = y_true >= 40.0
    m["mae_normal_ef"] = float(ae[mask_hi].mean()) if mask_hi.any() else float("nan")
    return m


def format_metrics(m: Dict[str, float]) -> str:
    return (
        f"MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | R2 {m['r2']:.4f} | "
        f"r {m['pearson']:.4f} | bias {m['bias']:+.3f} | "
        f"+-5pt {100 * m['agree_5']:.1f}%"
    )


def summarize_curve(history: list, key: str = "val_mae") -> Dict[str, float]:
    """Best epoch and its metric (the reported test score always uses the best model)."""
    if not history:
        return {}
    vals = [h.get(key, float("inf")) for h in history]
    best = int(np.argmin(vals))
    return {"best_epoch": best + 1, "best_val": float(vals[best])}
