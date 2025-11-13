# region imports
from AlgorithmImports import *
# endregion
from enum import Enum

import numpy as np


class RegimeAccordingToNATR(str, Enum):
    FOR_SURE_HIGH_VOLATILITY = "for_sure_high_volatility"
    LIKELY_HIGH_VOLATILITY = "likely_high_volatility"
    MAYBE_HIGH_VOLATILITY = "maybe_high_volatility"
    UNCLEAR = "unclear"


def classify_regime_according_to_natr(natr_value: float) -> RegimeAccordingToNATR:
    if natr_value > 2.6:
        return RegimeAccordingToNATR.FOR_SURE_HIGH_VOLATILITY
    elif natr_value > 2:
        return RegimeAccordingToNATR.LIKELY_HIGH_VOLATILITY
    elif natr_value > 1.5:
        return RegimeAccordingToNATR.MAYBE_HIGH_VOLATILITY
    else:
        return RegimeAccordingToNATR.UNCLEAR


def calculate_probablistic_regime_classification_according_to_natr(
    natr_value: float,
) -> float:

    augmented_natr_value = natr_value * 100

    k = 0.09
    x_0 = 205

    y = 1 / (1 + np.exp(-k * (augmented_natr_value - x_0)))

    return round(y, 5)
