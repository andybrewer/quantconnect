# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime
from .signals_common import Signal, SignalDecision, SignalDecisionType


class Is945Signal(Signal):

    def decide(self, current_time: datetime) -> SignalDecision:

        is_trading_time = (
            current_time.hour == 9
            and current_time.minute == 45
            and current_time.second == 0
        )

        return SignalDecision(
            decision_type=(
                SignalDecisionType.POSITIVE
                if is_trading_time
                else SignalDecisionType.NEGATIVE
            ),
            degree_of_certainty=1,
            relevant_data={"current time": current_time.strftime("%H:%M:%S")},
        )
