# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta, date
from ..features.consolidation_box import ConsolidationBox, OpeningRange
from parameters import algorithm_parameters
from ..features.opening_range import OpeningRangeFactory
from ..signals.are_the_last_x_candles_closing_inside_the_exit_area import (
    AreTheLastXCandlesClosingInsideTheExitAreaSignal,
)
from ..features.consolidation_box import ConsolidationBoxFactory

from ..features.consolidation_box import BoxBreakInsideClassification


from ..features.intraday_slope import IntradayTrend, IntradayTrendClassification

from ..signals.is_d1_trend_indicating_volatility_signal import (
    IsD1TrendIndicatingVolatilitySignal,
)

from ..signals.is_d1_trend_lh_indicating_volatility_signal import (
    IsD1TrendLHAndPercentileIndicatingVolatilitySignal,
)

from ..models.models import MyCandle

from ..signals.is_current_candle_in_the_exit_area_signal import (
    IsCurrentCandleInTheExitAreaSignal,
)
from ..signals.has_x_or_more_closing_hlods_until_y_signal import (
    HasXorMoreClosingHLODsUntilYSignal,
)
from ..signals.is_12_00_signal import Is1200Signal
from ..signals.are_last_2_intraday_trends_the_same_signal import (
    AreLast2IntradayTrendDirectionsTheSameSignal,
)
from ..signals.is_last_day_intraday_trend_downward_signal import (
    IsLastDayIntradayTrendDirectionDownwardSignal,
)
from ..signals.is_last_day_intraday_trend_upward_signal import (
    IsLastDayIntradayTrendDirectionUpwardSignal,
)
from ..signals.average_slope_percentile_is_bigger_than_threshold_signal import (
    AverageSlopePercentileIsBiggerThanThresholdSignal,
)
from ..signals.a_box_just_closed_up_or_down_signal import ABoxJustClosedUpOrDownSignal
from ..signals.a_box_just_closed_higher_or_lower_signal import (
    ABoxJustClosedHigherOrLowerSignal,
)
from ..signals.is_natr_indicating_volatility import (
    IsNATRIndicatingVolatilitySignal,
)
from ..signals.is_overnight_gap_indicating_volatility import (
    IsOvernightGapIndicatingVolatilitySignal,
)
from .trade_deciders_common import TradeDecider
from ..signals.signals_common import SignalDecisionType
from .trade_deciders_common import TradeDecision, TradeDecisionType


class HighVolatilityBoxBreakingTradeDecider(TradeDecider):
    def __init__(
        self,
        name: str,
        quantity_to_trade: float,
        requires_15_min_box_confirmation: bool,
        ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation: bool,
    ) -> None:
        super().__init__(name, quantity_to_trade)
        self.is_buying: bool | None = None
        self.is_selling: bool | None = None
        self.requires_15_min_box_confirmation: bool = requires_15_min_box_confirmation
        self.ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation = ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation
        self.waiting_for_15_min_box_confirmation: bool = False
        self.last_short_entry_price = None  # <-- ADD HERE
        self.last_long_entry_price = None   # <-- ADD HERE
        self.moment_of_pre_signal: datetime | None = None
        self.candle_of_pre_signal: MyCandle | None = None
        self.date_of_pre_signal: date | None = None  # <-- add this line
        self.confirmation_consolidation_box: ConsolidationBox | None = None
        self.open_trade_fill_price: float | None = None

    def _assign_pre_signal_once_per_day(self, current_candle: MyCandle, current_moment: datetime):
        if self.date_of_pre_signal != current_moment.date():
            self.moment_of_pre_signal = current_moment
            self.candle_of_pre_signal = current_candle
            self.date_of_pre_signal = current_moment.date()
            print(f"[PRE-SIGNAL SET] {current_moment.strftime('%Y-%m-%d %H:%M:%S')} — Price: {current_candle.close}")
        else:
            print(f"[PRE-SIGNAL REUSED] {current_moment.strftime('%Y-%m-%d %H:%M:%S')} — Using pre-signal from: {self.moment_of_pre_signal.strftime('%Y-%m-%d %H:%M:%S')}")
    
        self.waiting_for_15_min_box_confirmation = True

    def reset_daily_state(self):
        self.waiting_for_15_min_box_confirmation = False
        self.moment_of_pre_signal = None
        self.candle_of_pre_signal = None
        self.date_of_pre_signal = None
        self.confirmation_consolidation_box = None
        self.is_buying = None
        self.is_selling = None

    def decide(
        self,
        is_1200_signal: Is1200Signal,
        is_d0_overnight_gap_indicating_volatility_signal: IsOvernightGapIndicatingVolatilitySignal,
        is_d1_overnight_gap_indicating_volatility_signal: IsOvernightGapIndicatingVolatilitySignal,
        is_natr_indicating_volatility_signal: IsNATRIndicatingVolatilitySignal,
        a_box_just_closed_higher_signal: ABoxJustClosedHigherOrLowerSignal,
        a_box_just_closed_lower_signal: ABoxJustClosedHigherOrLowerSignal,
        a_15_min_box_just_closed_down_signal: ABoxJustClosedUpOrDownSignal,
        a_15_min_box_just_closed_up_signal: ABoxJustClosedUpOrDownSignal,
        average_slope_2_days_percentile_bigger_than_90_threshold_signal: AverageSlopePercentileIsBiggerThanThresholdSignal,
        average_slope_2_days_percentile_bigger_than_75_threshold_signal: AverageSlopePercentileIsBiggerThanThresholdSignal,
        average_slope_1_day_percentile_bigger_than_95_threshold_signal: AverageSlopePercentileIsBiggerThanThresholdSignal,
        are_last_2_intraday_trend_directions_the_same_signal: AreLast2IntradayTrendDirectionsTheSameSignal,
        is_last_intraday_trend_directions_upward_signal: IsLastDayIntradayTrendDirectionUpwardSignal,
        is_last_intraday_trend_directions_downward_signal: IsLastDayIntradayTrendDirectionDownwardSignal,
        has_x_closing_hlods_until_limit_signal: HasXorMoreClosingHLODsUntilYSignal,
        is_d1_trend_indicating_volatility_signal: IsD1TrendIndicatingVolatilitySignal,
        is_d1_trend_lh_and_percentile_indicating_volatility_signal: IsD1TrendLHAndPercentileIndicatingVolatilitySignal,
        # TODO: CONVERT THESE TO SIGNALS, OR DONT (according to what is convenient in the long term)
        d1_trend: IntradayTrend,
        d2_trend: IntradayTrend,
        d1_trend_lh: float,
        d2_trend_lh: float,
        d0_intraday_gap: float,
        d1_intraday_gap: float,
        two_day_average_gap: float,
        current_moment: datetime,
        current_candle: MyCandle,
        first_candle_of_the_day: MyCandle,
        moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far: list[datetime],
        moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far: list[datetime],
        consolidation_box_factories: list[ConsolidationBoxFactory],
        opening_range_factories_to_consider: list[OpeningRangeFactory],
        all_1_minute_candles_by_time: dict[datetime, MyCandle],
        all_daily_data: dict[datetime, MyCandle],
        last_exit_price: float | None = None,
    ) -> TradeDecision:

        assert a_box_just_closed_higher_signal.looking_for_high
        assert a_box_just_closed_lower_signal.looking_for_low

        is_towards_first_candle_on_long: bool = (
            current_candle.close < first_candle_of_the_day.open
        )
        is_towards_first_candle_on_short: bool = not is_towards_first_candle_on_long

        relevant_data = {
            "signals_used": [
                is_d0_overnight_gap_indicating_volatility_signal.name,
                is_d1_overnight_gap_indicating_volatility_signal.name,
                is_natr_indicating_volatility_signal.name,
                a_box_just_closed_higher_signal.name,
                a_box_just_closed_lower_signal.name,
                is_1200_signal.name,
                # average_slope_2_days_percentile_bigger_than_90_threshold_signal.name,
                # average_slope_2_days_percentile_bigger_than_75_threshold_signal.name,
                # average_slope_1_day_percentile_bigger_than_95_threshold_signal.name,
                # are_last_2_intraday_trend_directions_the_same_signal.name,
                # is_last_intraday_trend_directions_upward_signal.name,
                # is_last_intraday_trend_directions_downward_signal.name,
                has_x_closing_hlods_until_limit_signal.name,
                is_d1_trend_indicating_volatility_signal.name,
                is_d1_trend_lh_and_percentile_indicating_volatility_signal.name,
            ]
        }

        feature_data: dict[str, float] = {
            "D1 Trend Slope": d1_trend.slope,
            "D1 Trend Direction": d1_trend.classification.value,
            "D2 Trend Slope": d2_trend.slope,
            "D2 Trend Direction": d2_trend.classification.value,
            "D0 Intraday Gap": d0_intraday_gap,
            "D1 Intraday Gap": d1_intraday_gap,
            "2 Day Average Gap": two_day_average_gap,
            "D1 Trend LH": d1_trend_lh,
            "D2 Trend LH": d2_trend_lh,
        }

        has_x_closing_hlods_until_limit_signal_decision = has_x_closing_hlods_until_limit_signal.decide(
            moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far=moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
            moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far=moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
            current_time=current_moment,
            current_price=current_candle.close,
            opening_range_factories_to_consider=opening_range_factories_to_consider,
            percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal=algorithm_parameters.percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal
        )
        has_x_closing_hlods_until_limit_signal.most_recent_decision = has_x_closing_hlods_until_limit_signal_decision
        
        is_1200 = (
            is_1200_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if is_1200_signal.most_recent_decision
            else False
        )



        closing_hods_tells_us_to_buy = (
            has_x_closing_hlods_until_limit_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE_BUY
            if has_x_closing_hlods_until_limit_signal.most_recent_decision
            else False
        )
        closing_lods_tells_us_to_sell = (
            has_x_closing_hlods_until_limit_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE_SELL
            if has_x_closing_hlods_until_limit_signal.most_recent_decision
            else False
        )

        is_last_trend_upward = (
            is_last_intraday_trend_directions_upward_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if is_last_intraday_trend_directions_upward_signal.most_recent_decision
            else False
        )
        is_last_trend_downward = (
            is_last_intraday_trend_directions_downward_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if is_last_intraday_trend_directions_downward_signal.most_recent_decision
            else False
        )
        are_last_2_trends_the_same = (
            are_last_2_intraday_trend_directions_the_same_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if are_last_2_intraday_trend_directions_the_same_signal.most_recent_decision
            else False
        )

        are_last_2_days_upward = is_last_trend_upward and are_last_2_trends_the_same
        are_last_2_days_downward = is_last_trend_downward and are_last_2_trends_the_same

        is_last_day_slope_bigger_than_95 = (
            average_slope_1_day_percentile_bigger_than_95_threshold_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if average_slope_1_day_percentile_bigger_than_95_threshold_signal.most_recent_decision
            else False
        )

        are_last_2_days_slopes_average_bigger_than_90 = (
            average_slope_2_days_percentile_bigger_than_90_threshold_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if average_slope_2_days_percentile_bigger_than_90_threshold_signal.most_recent_decision
            else False
        )

        are_last_2_days_slopes_average_bigger_than_75 = (
            average_slope_2_days_percentile_bigger_than_75_threshold_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            if average_slope_2_days_percentile_bigger_than_75_threshold_signal.most_recent_decision
            else False
        )

        d0_overnight_gap_tells_us_to_trade = (
            is_d0_overnight_gap_indicating_volatility_signal.most_recent_decision
            is not None
            and is_d0_overnight_gap_indicating_volatility_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )
        d1_overnight_gap_tells_us_to_trade = (
            is_d1_overnight_gap_indicating_volatility_signal.most_recent_decision
            is not None
            and is_d1_overnight_gap_indicating_volatility_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )

        natr_tells_us_to_trade = (
            is_natr_indicating_volatility_signal.most_recent_decision is not None
            and is_natr_indicating_volatility_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )

        box_closing_high_tells_us_to_trade = (
            a_box_just_closed_higher_signal.most_recent_decision is not None
            and a_box_just_closed_higher_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE_BUY
        )
        box_closing_low_tells_us_to_trade = (
            a_box_just_closed_lower_signal.most_recent_decision is not None
            and a_box_just_closed_lower_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE_SELL
        )

            
        a_15_min_box_closing_high_tells_us_to_trade = (
            a_15_min_box_just_closed_up_signal.most_recent_decision is not None
            and a_15_min_box_just_closed_up_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )
        a_15_min_box_closing_low_tells_us_to_trade = (
            a_15_min_box_just_closed_down_signal.most_recent_decision is not None
            and a_15_min_box_just_closed_down_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )

        is_d1_trend_indicating_volatility = (
            is_d1_trend_indicating_volatility_signal.most_recent_decision is not None
            and is_d1_trend_indicating_volatility_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )
        is_d1_trend_lh_and_percentile_indicating_volatility = (
            is_d1_trend_lh_and_percentile_indicating_volatility_signal.most_recent_decision
            is not None
            and is_d1_trend_lh_and_percentile_indicating_volatility_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        )

        # Define high volatility now (must be before using it below)
        is_high_volatility = (
            d0_overnight_gap_tells_us_to_trade is True
            or d1_overnight_gap_tells_us_to_trade is True
            or is_d1_trend_indicating_volatility
            or is_d1_trend_lh_and_percentile_indicating_volatility
        )

        balance_tells_us_to_trade = closing_hods_tells_us_to_buy or closing_lods_tells_us_to_sell

        if (
            is_high_volatility
            and self.date_of_pre_signal != current_moment.date()
            and (
                box_closing_high_tells_us_to_trade
                or box_closing_low_tells_us_to_trade
                or balance_tells_us_to_trade
            )
        ):
            self._assign_pre_signal_once_per_day(current_candle, current_moment)

            if self.requires_15_min_box_confirmation:
                return TradeDecision(
                    decision_type=TradeDecisionType.DONT_TRADE,
                    order_quantity=self.quantity_to_trade,
                    relevant_data={"note": "Pre-signal set due to balance breakout or HOD/LOD signal"},
                    is_opposite=False,
                )
            
    

        # TODO: DELETE THIS, MEANT FOR TESTING PURPOSES ONLY
        # if is_1200:
        #     return TradeDecision(
        #         decision_type=TradeDecisionType.BUY,
        #         order_quantity=self.quantity_to_trade,
        #         relevant_data=relevant_data,
        #         is_opposite=False,
        #     )
        # END OF DELETING PART

        # AVOID PARAMETERS FOR THE CONFIRMATION CONS BOX
        should_we_avoid_a_short_trade_confirmation_box = False
        should_we_avoid_a_long_trade_confirmation_box = False
        should_we_avoid_a_short_trade_confirmation_box = any(
            [
                (   d0_intraday_gap >= -0.0015 and d0_intraday_gap <= 0.0015 and d1_trend_lh >= 0.0075
                ),
                (   d1_trend.classification == IntradayTrendClassification.DOWNWARD 
                ),
                (   d1_trend.classification == IntradayTrendClassification.UPWARD and d0_intraday_gap >= 0 and d2_trend.slope <= 0
                ),
               
               
                
            ]
        )

        should_we_avoid_a_long_trade_confirmation_box = any(
            [
                (   d0_intraday_gap >= -0.0015 and d0_intraday_gap <= 0.0015 and d1_trend_lh >= 0.0075
                ),
                (   d1_trend.classification == IntradayTrendClassification.UPWARD
                ),
                (   d1_trend.classification == IntradayTrendClassification.DOWNWARD 
                ),
               
              
            ]
        )

        if self.moment_of_pre_signal and self.date_of_pre_signal == current_moment.date():
            for box_factory in consolidation_box_factories:
                for cons_box in box_factory.closed_consolidation_boxes:
                    if cons_box.end_moment != current_moment - timedelta(minutes=1):
                        continue

                    box_price_reference = all_daily_data[cons_box.end_moment].close  # actual breakout close

                    # --- Reject boxes too far from pre-signal close ---
                    max_dist_from_pre_signal = algorithm_parameters.max_percent_distance_from_pre_signal_close_to_box
                    pre_close = self.candle_of_pre_signal.close
                    too_far_from_pre_close, dist_from_pre = self.is_price_too_far_from_reference(
                        current_price=box_price_reference,
                        reference_price=pre_close,
                        max_percent=max_dist_from_pre_signal
                    )
                    if too_far_from_pre_close:
                        print(f"[BOX TOO FAR FROM PRE] Skipping confirmation box at {cons_box.end_moment.strftime('%H:%M')} — "
                              f"Distance: {dist_from_pre:.2f}% > {max_dist_from_pre_signal:.2f}%")
                        continue

                    # --- Reject boxes too far from last exit price ---
                    max_dist_from_last_exit = algorithm_parameters.max_percent_distance_from_last_exit_price_to_box
                    if last_exit_price is not None:
                        too_far_from_exit, dist_from_exit = self.is_price_too_far_from_reference(
                            current_price=box_price_reference,
                            reference_price=last_exit_price,
                            max_percent=max_dist_from_last_exit
                        )
                        if too_far_from_exit:
                            print(f"[BOX TOO FAR FROM EXIT] Skipping confirmation box at {cons_box.end_moment.strftime('%H:%M')} — "
                                f"Distance: {dist_from_exit:.2f}% > {max_dist_from_last_exit:.2f}%")
                            continue

                    # --- Check box timing relative to pre-signal ---
                    is_valid_box_timing = cons_box.start_moment >= self.moment_of_pre_signal - timedelta(
                        minutes=algorithm_parameters.allowed_minutes_overlap_of_start_of_confirmation_box_and_pre_signal_moment
                    )
                    if not is_valid_box_timing:
                        continue

                    # --- Debug output ---
                    print(
                        f"[CONFIRMATION CHECK] {current_moment.strftime('%Y-%m-%d %H:%M')} | "
                        f"pre-signal: {self.moment_of_pre_signal.strftime('%H:%M')} | "
                        f"box_start: {cons_box.start_moment.strftime('%H:%M')} | "
                        f"box_high: {cons_box.high:.2f} | "
                        f"box_low: {cons_box.low:.2f} | "
                        f"pre_close: {pre_close:.2f} | "
                        f"now_close: {current_candle.close:.2f} | "
                        f"box_valid_start: {is_valid_box_timing}"
                    )

                    # ─── 3-min avg-close midpoint filter ───
                    start_time = cons_box.end_moment - timedelta(minutes=3)
                    closes = []
                    for i in range(3):
                        ts = start_time + timedelta(minutes=i)
                        candle = all_1_minute_candles_by_time.get(ts)
                        if candle:
                            closes.append(candle.close)

                    if closes:
                        avg_close = sum(closes) / len(closes)
                        midpoint  = (cons_box.high + cons_box.low) / 2
                        # if this was tagged as an up-break, but avg_close was at or below midpoint → reject
                        if cons_box.box_break_inside == BoxBreakInsideClassification.UP and avg_close <= midpoint * 0.95:
                            print(f"[MIDPOINT FILTER] Reject UP break at {cons_box.end_moment.strftime('%H:%M')} "
                                  f"(avg_close {avg_close:.2f} ≤ midpoint {midpoint:.2f})")
                            continue
                        # if this was tagged as a down-break, but avg_close was at or above midpoint → reject
                        if cons_box.box_break_inside == BoxBreakInsideClassification.DOWN and avg_close >= midpoint * 1.05:
                            print(f"[MIDPOINT FILTER] Reject DOWN break at {cons_box.end_moment.strftime('%H:%M')} "
                                  f"(avg_close {avg_close:.2f} ≥ midpoint {midpoint:.2f})")
                            continue
                    # ─── end avg-close midpoint filter ───

                    # Enforce midpoint-directional bias filter before accepting the confirmation box
                    box_midpoint = (cons_box.high + cons_box.low) / 2
                    total_candles = 0
                    closes_above = 0
                    closes_below = 0

                    for i in range(int((cons_box.end_moment - cons_box.start_moment).total_seconds() / 60) + 1):
                        moment = cons_box.start_moment + timedelta(minutes=i)
                        if moment not in all_1_minute_candles_by_time:
                            continue
                        close = all_1_minute_candles_by_time[moment].close
                        total_candles += 1
                        if close > box_midpoint:
                            closes_above += 1
                        elif close < box_midpoint:
                            closes_below += 1

                    if cons_box.box_break_inside == BoxBreakInsideClassification.UP and closes_above < total_candles / 10:
                        print(f"[REJECTED] Box broke UP but only {closes_above}/{total_candles} closes above midpoint")
                        continue

                    if cons_box.box_break_inside == BoxBreakInsideClassification.DOWN and closes_below < total_candles / 10:
                        print(f"[REJECTED] Box broke DOWN but only {closes_below}/{total_candles} closes below midpoint")
                        continue

                    height_of_box = cons_box.high - cons_box.low
                    is_sufficiently_far_away_to_go_long = current_candle.close >= self.candle_of_pre_signal.close + (
                        height_of_box * algorithm_parameters.coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_above
                    )
                    is_sufficiently_far_away_to_go_short = current_candle.close <= self.candle_of_pre_signal.close - (
                        height_of_box * algorithm_parameters.coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_below
                    )

                    if cons_box.box_break_inside == BoxBreakInsideClassification.DOWN:
                        if not should_we_avoid_a_short_trade_confirmation_box and is_sufficiently_far_away_to_go_short:
                            if hasattr(self, "last_short_entry_price") and self.last_short_entry_price is not None:
                                relative_dist = abs(current_candle.close - self.last_short_entry_price) / self.last_short_entry_price
                                if (
                                    relative_dist <= 0.001
                                    or current_candle.close > self.last_short_entry_price
                                ):
                                    print(f"[BLOCKED] Short entry blocked: current={current_candle.close:.2f}, last={self.last_short_entry_price:.2f}")
                                    return TradeDecision(
                                        decision_type=TradeDecisionType.DONT_TRADE,
                                        order_quantity=self.quantity_to_trade,
                                        relevant_data={"note": "Short too close or above last short"},
                                        is_opposite=False,
                                    )

                            self.confirmation_consolidation_box = cons_box
                            self.waiting_for_15_min_box_confirmation = False
                            self.is_selling = True
                            self.last_short_entry_price = current_candle.close
                            return TradeDecision(
                                decision_type=TradeDecisionType.SELL,
                                order_quantity=self.quantity_to_trade,
                                relevant_data={"note": "Box broke down and passed filters"},
                                is_opposite=False,
                                pre_signal_moment=self.moment_of_pre_signal,
                                is_towards_opening_price=is_towards_first_candle_on_short,
                            )

                    elif cons_box.box_break_inside == BoxBreakInsideClassification.UP:
                        if not should_we_avoid_a_long_trade_confirmation_box and is_sufficiently_far_away_to_go_long:
                            if hasattr(self, "last_long_entry_price") and self.last_long_entry_price is not None:
                                relative_dist = abs(current_candle.close - self.last_long_entry_price) / self.last_long_entry_price
                                if relative_dist <= 0.001:
                                    print(f"[BLOCKED] Long entry too close to last long: current={current_candle.close:.2f}, last={self.last_long_entry_price:.2f}")
                                    return TradeDecision(
                                        decision_type=TradeDecisionType.DONT_TRADE,
                                        order_quantity=self.quantity_to_trade,
                                        relevant_data={"note": "Long too close to last long"},
                                        is_opposite=False,
                                    )

                            self.confirmation_consolidation_box = cons_box
                            self.waiting_for_15_min_box_confirmation = False
                            self.is_buying = True
                            self.last_long_entry_price = current_candle.close
                            return TradeDecision(
                                decision_type=TradeDecisionType.BUY,
                                order_quantity=self.quantity_to_trade,
                                relevant_data={"note": "Box broke up and passed filters"},
                                is_opposite=False,
                                pre_signal_moment=self.moment_of_pre_signal,
                                is_towards_opening_price=is_towards_first_candle_on_long,
                            )
                        
        is_high_volatility = (
            d0_overnight_gap_tells_us_to_trade is True
            or d1_overnight_gap_tells_us_to_trade is True
            # or natr_tells_us_to_trade is True
            or is_d1_trend_indicating_volatility
            or is_d1_trend_lh_and_percentile_indicating_volatility
        )

        should_we_avoid_a_long_trade = any(
            [
                
            ]
        )
        if (
            self.ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation
        ):
            should_we_avoid_a_long_trade = False

        should_we_go_opposite_of_long = any(

        [
                
        ]
           
        )

        should_we_avoid_a_short_trade = any(
            [
               
            ]
        )
        if (
            self.ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation
        ):
            should_we_avoid_a_short_trade = False

        should_we_go_opposite_of_short = any(

            [
                

            
            ]
        )

        if not is_high_volatility and not (box_closing_high_tells_us_to_trade or box_closing_low_tells_us_to_trade):
            return TradeDecision(
                decision_type=TradeDecisionType.DONT_TRADE,
                order_quantity=self.quantity_to_trade,
                relevant_data=relevant_data,
                feature_data=feature_data,
                is_opposite=False,
            )

        # --- Global filter: distance from last exit price ---
        if hasattr(self, "last_exit_price") and self.last_exit_price is not None:
            distance_from_exit = abs(current_candle.close - self.last_exit_price) / self.last_exit_price * 100
            if distance_from_exit > algorithm_parameters.max_percent_distance_from_last_exit_price_to_box:
                print(f"[ENTRY TOO FAR FROM LAST EXIT] Rejecting trade at {current_moment.strftime('%H:%M')} — "
                      f"Distance: {distance_from_exit:.2f}% > "
                      f"{algorithm_parameters.max_percent_distance_from_last_exit_price_to_box:.2f}%")
                return TradeDecision.dont_trade(reason="Too far from last exit")
            
        if box_closing_high_tells_us_to_trade:

            # now we check for specific reasons not to go LONG
            relevant_data["signals_used"].extend(
                [
                    average_slope_2_days_percentile_bigger_than_90_threshold_signal.name,
                    are_last_2_intraday_trend_directions_the_same_signal.name,
                    average_slope_1_day_percentile_bigger_than_95_threshold_signal.name,
                    is_last_intraday_trend_directions_upward_signal.name,
                ]
            )
            if not should_we_avoid_a_long_trade:

                if self.requires_15_min_box_confirmation:
                    self._assign_pre_signal_once_per_day(current_candle, current_moment)
                else:
                    if should_we_go_opposite_of_long:
                        self.is_selling = True
                        return TradeDecision(
                            decision_type=TradeDecisionType.SELL,
                            order_quantity=self.quantity_to_trade,
                            relevant_data=relevant_data,
                            feature_data=feature_data,
                            is_opposite=True,
                            is_towards_opening_price=is_towards_first_candle_on_short,
                        )
                    else:
                        self.is_buying = True
                        return TradeDecision(
                            decision_type=TradeDecisionType.BUY,
                            order_quantity=self.quantity_to_trade,
                            relevant_data=relevant_data,
                            feature_data=feature_data,
                            is_opposite=False,
                            is_towards_opening_price=is_towards_first_candle_on_long,
                        )
        elif box_closing_low_tells_us_to_trade:

            # now we check for specific reasons not to go SHORT
            relevant_data["signals_used"].extend(
                [
                    average_slope_2_days_percentile_bigger_than_90_threshold_signal.name,
                    are_last_2_intraday_trend_directions_the_same_signal.name,
                    average_slope_1_day_percentile_bigger_than_95_threshold_signal.name,
                    is_last_intraday_trend_directions_downward_signal.name,
                ]
            )
            if not should_we_avoid_a_short_trade:
                if self.requires_15_min_box_confirmation:
                    self._assign_pre_signal_once_per_day(current_candle, current_moment)

                else:
                    if should_we_go_opposite_of_short:
                        self.is_buying = True
                        return TradeDecision(
                            decision_type=TradeDecisionType.BUY,
                            order_quantity=self.quantity_to_trade,
                            relevant_data=relevant_data,
                            feature_data=feature_data,
                            is_opposite=True,
                            is_towards_opening_price=is_towards_first_candle_on_long,
                        )
                    else:
                        self.is_selling = True
                        return TradeDecision(
                            decision_type=TradeDecisionType.SELL,
                            order_quantity=self.quantity_to_trade,
                            relevant_data=relevant_data,
                            feature_data=feature_data,
                            is_opposite=False,
                            is_towards_opening_price=is_towards_first_candle_on_short,
                        )

        return TradeDecision(
            decision_type=TradeDecisionType.DONT_TRADE,
            order_quantity=self.quantity_to_trade,
            relevant_data=relevant_data,
            feature_data=feature_data,
            is_opposite=False,
        )
    
    def is_price_too_far_from_reference(self, current_price: float, reference_price: float, max_percent: float) -> tuple[bool, float]:
        if reference_price is None:
            return False, 0.0
        distance = abs(current_price - reference_price) / reference_price * 100
        return distance > max_percent, distance

    def get_take_profit_initial_value(
        self,
        candle: MyCandle,
        current_opening_range: OpeningRange,
        coeficient_of_box_height: float | None = None,
        coeficient_of_opening_range: float | None = None,
    ) -> tuple[float, float] | None:
        if not self.is_buying and not self.is_selling:
            return None
        if self.requires_15_min_box_confirmation is False:
            assert coeficient_of_opening_range
            distance_from_current_candle = (
                current_opening_range.high - current_opening_range.low
            ) * coeficient_of_opening_range
            if self.is_buying:
                return (
                    candle.close + distance_from_current_candle,
                    distance_from_current_candle,
                )
            else:
                return (
                    candle.close - distance_from_current_candle,
                    distance_from_current_candle,
                )
        assert self.confirmation_consolidation_box
        assert coeficient_of_box_height
        distance_from_current_candle = (
            self.confirmation_consolidation_box.high
            - self.confirmation_consolidation_box.low
        )
        if self.is_buying:
            return (
                self.confirmation_consolidation_box.high
                + distance_from_current_candle * coeficient_of_box_height
            ), distance_from_current_candle
        else:
            return (
                self.confirmation_consolidation_box.low
                - distance_from_current_candle * coeficient_of_box_height
            ), distance_from_current_candle

    def should_exit_trade(
        self,
        is_market_open: bool,
        are_last_x_candles_closing_inside_the_exit_area_signal: AreTheLastXCandlesClosingInsideTheExitAreaSignal,
        is_current_candle_in_the_exit_area_signal: IsCurrentCandleInTheExitAreaSignal,
        a_15_min_box_just_closed_up_signal: ABoxJustClosedUpOrDownSignal,
        a_15_min_box_just_closed_down_signal: ABoxJustClosedUpOrDownSignal,
        current_candle: MyCandle,
        is_current_candle_in_the_take_profit_area: bool,
    ) -> None | bool:
        assert a_15_min_box_just_closed_up_signal.looking_for_up
        assert a_15_min_box_just_closed_down_signal.looking_for_down
        if self.open_trade is None:
            return None

        if is_market_open is False:
            return True

        # did_it_just_close_up = False
        # if (
        #     a_15_min_box_just_closed_up_signal.most_recent_decision
        #     and a_15_min_box_just_closed_up_signal.most_recent_decision.decision_type
        #     == SignalDecisionType.POSITIVE
        # ):
        #     did_it_just_close_up = True

        # did_it_just_close_down = False
        # if (
        #     a_15_min_box_just_closed_down_signal.most_recent_decision
        #     and a_15_min_box_just_closed_down_signal.most_recent_decision.decision_type
        #     == SignalDecisionType.POSITIVE
        # ):
        #     did_it_just_close_down = True

        # price_at_moment_of_order = self.open_trade.average_fill_price
        # are_we_profiting_with_our_open_trade = (
        #     current_candle.close >= price_at_moment_of_order
        #     if self.open_trade.quantity > 0
        #     else current_candle.close <= price_at_moment_of_order
        # )

        if is_current_candle_in_the_take_profit_area:
            return True

        if (
            is_current_candle_in_the_exit_area_signal.most_recent_decision
            and is_current_candle_in_the_exit_area_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        ) and (
            are_last_x_candles_closing_inside_the_exit_area_signal.most_recent_decision
            and are_last_x_candles_closing_inside_the_exit_area_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
        ):
            return True

        return False
