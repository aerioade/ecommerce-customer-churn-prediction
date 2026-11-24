"""Map positive-class churn probabilities to retention recommendations."""

import numpy as np
import pandas as pd

from src.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

# The binary prediction and the business risk tier serve different purposes.
CHURN_CLASS_THRESHOLD = 0.50

RISK_DETAILS = {
    "Low Risk": {
        "color": "#16A34A",
        "indication": "Prioritas retensi rendah.",
        "action": "Pertahankan loyalitas dengan poin, program VIP, atau akses awal produk.",
    },
    "Medium Risk": {
        "color": "#D97706",
        "indication": "Pelanggan perlu didorong untuk kembali berbelanja.",
        "action": "Kirim notifikasi kategori favorit dan tawarkan cashback terarah.",
    },
    "High Risk": {
        "color": "#DC2626",
        "indication": "Pelanggan perlu diprioritaskan untuk retensi.",
        "action": "Hubungi pelanggan, tangani komplain dalam 24 jam, dan tawarkan voucher retensi.",
    },
}


def _risk_tiers(probabilities):
    """Validate probabilities and classify them using the configured cutoffs."""
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Probabilitas churn harus berupa angka finite antara 0 dan 1.")
    return np.select(
        [probabilities >= HIGH_RISK_THRESHOLD, probabilities >= MEDIUM_RISK_THRESHOLD],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )


def classify_risk(probability: float) -> dict[str, str]:
    """Return the risk tier, badge color, indication, and retention action."""
    value = np.asarray(probability, dtype=float)
    if value.ndim != 0:
        raise ValueError("Prediksi individual membutuhkan satu probabilitas churn.")
    tier = _risk_tiers(value).item()
    return {"tier": tier, **RISK_DETAILS[tier]}


def apply_risk_to_batch(customers: pd.DataFrame, probabilities) -> pd.DataFrame:
    """Append predictions and risk details using one probability per row in order."""
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(customers),):
        raise ValueError("Diperlukan satu probabilitas churn per baris pelanggan.")
    tiers = _risk_tiers(values)

    result = customers.copy()
    result["Churn_Probability"] = values
    result["Predicted_Churn"] = (values >= CHURN_CLASS_THRESHOLD).astype(int)
    result["Risk_Tier"] = tiers
    actions = {tier: details["action"] for tier, details in RISK_DETAILS.items()}
    result["Recommended_Action"] = result["Risk_Tier"].map(actions)
    return result
