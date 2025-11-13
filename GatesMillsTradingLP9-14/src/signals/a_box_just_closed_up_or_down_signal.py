# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta
from ..features.consolidation_box import (
    BoxBreakInsideClassification,
    ConsolidationBoxFactory,
)
from .signals_common import Signal, SignalDecision, SignalDecisionType


class ABoxJustClosedUpOrDownSignal(Signal):
    def __init__(
        self,
        name: str,
        looking_for_up: bool,
        looking_for_down: bool,
    ) -> None:
        super().__init__(name=name)
        assert looking_for_up + looking_for_down == 1
        self.looking_for_down = looking_for_down
        self.looking_for_up = looking_for_up

    def decide(
        self,
        current_datetime: datetime,
        consolidation_box_factories: list[ConsolidationBoxFactory],
    ) -> SignalDecision:

        box_break_inside_goal = (
            BoxBreakInsideClassification.UP
            if self.looking_for_up
            else BoxBreakInsideClassification.DOWN
        )

        for box_factory in consolidation_box_factories:
            for cons_box in box_factory.closed_consolidation_boxes:
                if (
                    current_datetime - timedelta(minutes=1) == cons_box.end_moment
                    and cons_box.box_break_inside == box_break_inside_goal
                ):
                    return SignalDecision(
                        decision_type=SignalDecisionType.POSITIVE,
                        degree_of_certainty=1,
                        relevant_data={"consolidation_box": cons_box.model_dump()},
                    )

        return SignalDecision(
            decision_type=SignalDecisionType.NEGATIVE,
            degree_of_certainty=1,
            relevant_data={},
        )
