# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime
from ..models.models import MyCandle
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsCurrentCandleInTheExitAreaSignal(Signal):
    def decide(
        self,
        current_candle: MyCandle,
        current_moment: datetime,
        in_exit_mode_of_buy: bool,
        in_exit_mode_of_sell: bool,
        exit_area_when_buying: dict[datetime, float],
        exit_area_when_selling: dict[datetime, float],
    ) -> SignalDecision:

        assert in_exit_mode_of_buy + in_exit_mode_of_sell <= 1

        if in_exit_mode_of_buy:
            if (
                current_moment in exit_area_when_buying.keys()
                and current_candle.close <= exit_area_when_buying[current_moment]
            ):
                return SignalDecision(
                    decision_type=SignalDecisionType.POSITIVE,
                    degree_of_certainty=1,
                    relevant_data={
                        "exit area boundary": exit_area_when_buying[current_moment],
                        "stepped into exit area with closing value": current_candle.close,
                        "in exit of buy mode": in_exit_mode_of_buy,
                        "in exit of sell mode": in_exit_mode_of_sell,
                    },
                )
        if in_exit_mode_of_sell:
            if (
                current_moment in exit_area_when_selling.keys()
                and current_candle.close >= exit_area_when_selling[current_moment]
            ):
                return SignalDecision(
                    decision_type=SignalDecisionType.POSITIVE,
                    degree_of_certainty=1,
                    relevant_data={
                        "exit area boundary": exit_area_when_selling[current_moment],
                        "stepped into exit area with closing value": current_candle.close,
                        "in exit of buy mode": in_exit_mode_of_buy,
                        "in exit of sell mode": in_exit_mode_of_sell,
                    },
                )

        return SignalDecision(
            decision_type=SignalDecisionType.NEGATIVE,
            degree_of_certainty=1,
            relevant_data={
                "in exit of buy mode": in_exit_mode_of_buy,
                "in exit of sell mode": in_exit_mode_of_sell,
            },
        )
