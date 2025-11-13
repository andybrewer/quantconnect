import copy
from datetime import date, datetime, timedelta
import math
import json
import time as time_module

from QuantConnect.Data.Market import FuturesContract, TradeBar
from parameters import algorithm_parameters
from AlgorithmImports import *
from QuantConnect.Securities.Equity import Equity
from QuantConnect.Orders import OrderEvent, OrderStatus, OrderDirection


import numpy as np
from scipy.stats import percentileofscore
from src.features.dx_trend import (
    dx_trend_feature,
    get_closing_price_from_knowing_trend_and_open_price,
)
from src.features.dx_trend_lh import dx_trend_lh_feature
from src.features.dx_gap_magnitude import (
    OvernightGapClassification,
    classify_dx_gap_magnitude,
    dx_gap_magnitude_feature,
    gap_magnitude_2_day_average_feature,
)
from src.features.intraday_slope import (
    IntradayTrendClassification,
    dx_intraday_trend_slope,
    IntradayTrend,
)
from src.features.consolidation_box import BoxBreakInsideClassification

from src.features.rolling_standard_deviation import rolling_standard_deviation_feature
from src.features.consolidation_box import (
    ConsolidationBoxFactory,
)
from src.features.opening_range import (
    OpeningRange,
    OpeningRangeFactory,
    get_average_opening_range_thickness,
)
from src.signals.signals_common import SignalDecision, SignalDecisionType
from src.models.models import MyCandle, TakeProfitArea

from src.signals.is_9_45_signal import Is945Signal
from src.signals.is_12_00_signal import Is1200Signal
from src.signals.is_current_candle_away_from_the_open_value_signal import (
    IsCurrentCandleAwayFromTheOpenValueSignal,
)
from src.trade_deciders.trade_deciders_common import (
    TradeDecider,
    TradeDecision,
    TradeDecisionType,
)
from src.signals.last_consolidation_box_closed_at_a_daily_high_or_low_signal import (
    LastConsolidationBoxClosedAtADailyHighOrLowSignal,
)
from src.features.regime_according_to_natr import (
    RegimeAccordingToNATR,
    calculate_probablistic_regime_classification_according_to_natr,
    classify_regime_according_to_natr,
)
from src.signals.a_box_just_closed_higher_or_lower_signal import (
    ABoxJustClosedHigherOrLowerSignal,
)
from src.signals.is_natr_indicating_volatility import (
    IsNATRIndicatingVolatilitySignal,
)
from src.signals.is_overnight_gap_indicating_volatility import (
    IsOvernightGapIndicatingVolatilitySignal,
)
from src.trade_deciders.high_volatility_box_breaking_decider import (
    HighVolatilityBoxBreakingTradeDecider,
)
from src.signals.a_box_just_closed_up_or_down_signal import ABoxJustClosedUpOrDownSignal
from src.signals.average_slope_percentile_is_bigger_than_threshold_signal import (
    AverageSlopePercentileIsBiggerThanThresholdSignal,
)
from src.signals.are_last_2_intraday_trends_the_same_signal import (
    AreLast2IntradayTrendDirectionsTheSameSignal,
)
from src.signals.is_last_day_intraday_trend_upward_signal import (
    IsLastDayIntradayTrendDirectionUpwardSignal,
)
from src.signals.is_last_day_intraday_trend_downward_signal import (
    IsLastDayIntradayTrendDirectionDownwardSignal,
)
from src.signals.has_x_or_more_closing_hlods_until_y_signal import (
    HasXorMoreClosingHLODsUntilYSignal,
)
from src.signals.is_current_candle_in_the_exit_area_signal import (
    IsCurrentCandleInTheExitAreaSignal,
)
from src.signals.is_d1_trend_indicating_volatility_signal import (
    IsD1TrendIndicatingVolatilitySignal,
)
from src.signals.is_d1_trend_lh_indicating_volatility_signal import (
    IsD1TrendLHAndPercentileIndicatingVolatilitySignal,
)
from src.signals.are_the_last_x_candles_closing_inside_the_exit_area import (
    AreTheLastXCandlesClosingInsideTheExitAreaSignal,
)
from pydantic import BaseModel

STRFTIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ACCOUNT_DELAY_SECONDS = 15

class FuturesTradingParameters(BaseModel):
    is_observing_futures: bool
    is_trading_futures: bool
    contracts_minimum_expiry_days: int
    contracts_maximum_expiry_days: int


futures_trading_parameters = FuturesTradingParameters(
    is_observing_futures=False,
    is_trading_futures=False,
    contracts_minimum_expiry_days=0,
    contracts_maximum_expiry_days=200,
)


class HolyGrailProject(QCAlgorithm):

    def initialize(self):
        self.enable_logs: bool = True

        self.number_of_trades = 0
        self.number_of_exits = 0

        self.set_start_date(
            algorithm_parameters.start_date.year,
            algorithm_parameters.start_date.month,
            algorithm_parameters.start_date.day,
        )  # Set Start Date
        self.set_end_date(
            algorithm_parameters.end_date.year,
            algorithm_parameters.end_date.month,
            algorithm_parameters.end_date.day,
        )  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash

        self.set_warm_up(timedelta(days=1 * 30, hours=0))

        self.days_to_ignore = [
            datetime(2013, 10, 3).date(),
            datetime(2013, 10, 4).date(),
            datetime(2013, 10, 7).date(),
            datetime(2013, 10, 9).date(),
            datetime(2013, 10, 10).date(),
            datetime(2013, 10, 11).date(),
            datetime(2018, 7, 27).date(),
            datetime(2018, 9, 17).date(),
            datetime(2019, 8, 5).date(),
            datetime(2019, 9, 13).date(),
            datetime(2020, 3, 9).date(),
            datetime(2020, 3, 12).date(),
            datetime(2020, 3, 16).date(),
            datetime(2020, 3, 18).date(),
            datetime(2008, 7, 3).date(),
            datetime(2009, 7, 3).date(),
            datetime(2010, 7, 2).date(),
            datetime(2011, 7, 1).date(),
            datetime(2012, 7, 3).date(),
            datetime(2013, 7, 3).date(),
            datetime(2014, 7, 3).date(),
            datetime(2015, 7, 3).date(),
            datetime(2016, 7, 1).date(),
            datetime(2017, 7, 3).date(),
            datetime(2018, 7, 3).date(),
            datetime(2019, 7, 3).date(),
            datetime(2020, 7, 3).date(),
            datetime(2021, 7, 2).date(),
            datetime(2022, 7, 4).date(),
            datetime(2023, 7, 3).date(),
            datetime(2024, 7, 3).date(),
            datetime(2008, 11, 28).date(),
            datetime(2009, 11, 27).date(),
            datetime(2010, 11, 26).date(),
            datetime(2011, 11, 25).date(),
            datetime(2012, 11, 23).date(),
            datetime(2013, 11, 29).date(),
            datetime(2014, 11, 28).date(),
            datetime(2015, 11, 27).date(),
            datetime(2016, 11, 25).date(),
            datetime(2017, 11, 24).date(),
            datetime(2018, 11, 23).date(),
            datetime(2019, 11, 29).date(),
            datetime(2020, 11, 27).date(),
            datetime(2021, 11, 26).date(),
            datetime(2022, 11, 25).date(),
            datetime(2023, 11, 24).date(),
            datetime(2024, 11, 29).date(),
            datetime(2008, 12, 24).date(),
            datetime(2009, 12, 24).date(),
            datetime(2010, 12, 24).date(),
            datetime(2011, 12, 23).date(),
            datetime(2012, 12, 24).date(),
            datetime(2013, 12, 24).date(),
            datetime(2014, 12, 24).date(),
            datetime(2015, 12, 24).date(),
            datetime(2016, 12, 23).date(),
            datetime(2017, 12, 22).date(),
            datetime(2018, 12, 24).date(),
            datetime(2019, 12, 24).date(),
            datetime(2020, 12, 24).date(),
            datetime(2021, 12, 24).date(),
            datetime(2022, 12, 23).date(),
            datetime(2023, 12, 22).date(),
            datetime(2024, 12, 24).date(),

        ]

        current_resolution = Resolution.MINUTE

        self.current_ticker: str = "QQQ"

        assert algorithm_parameters.proportion_of_cash_to_trade >= 0

        self.current_ticker_object: Equity = self.add_equity(
            self.current_ticker,
            current_resolution,
            data_normalization_mode=DataNormalizationMode.RAW,
        )

        self.current_symbol = self.current_ticker_object.symbol

        # ── Leverage config ─────────────────────────────────────────────
        # Desired leverage for sizing
        self.target_leverage = getattr(algorithm_parameters, "target_leverage", 1)  # e.g., 3×

        # Max leverage engine will allow (must be ≥ target_leverage)
        self.leverage_cap = getattr(algorithm_parameters, "leverage_cap", 5.0)

        # Apply leverage cap in QC
        self.UniverseSettings.Leverage = self.leverage_cap
        self.Securities[self.current_symbol].SetLeverage(self.leverage_cap)
        # ────────────────────────────────────────────────────────────────

        if futures_trading_parameters.is_trading_futures:
            self._future = self.add_future(
                Futures.Indices.NASDAQ100EMini,
                extended_market_hours=True,
                resolution=current_resolution,
            )
            if futures_trading_parameters.is_observing_futures:
                self.current_symbol = self._future.symbol
            self._future.set_filter(
                futures_trading_parameters.contracts_minimum_expiry_days,
                futures_trading_parameters.contracts_maximum_expiry_days,
            )

            self.selected_futures_contract = None

        self.market_was_ever_open = False

        self.schedule.on(
            self.date_rules.every_day(self.current_ticker),
            self.time_rules.after_market_open(self.current_ticker),
            self.market_open_callback,
        )

        self.schedule.on(
            self.date_rules.every_day(self.current_ticker),
            self.time_rules.before_market_close(self.current_ticker),
            self.market_close_callback,
        )

        self.last_market_open_moment: datetime | None = None

        self.has_open_trade: bool = False
        self.open_trade_quantity: float | None = None
        self.open_trade_fill_price = None

        self.market_open_prices_by_date: dict[date, float] = {}
        self.market_close_prices_by_date: dict[date, float] = {}

        self.market_high_by_date: dict[date, float] = {}
        self.market_low_by_date: dict[date, float] = {}

        self.market_high_moment_by_date: dict[date, datetime] = {}
        self.market_low_moment_by_date: dict[date, datetime] = {}

        self.all_daily_data: dict[datetime, MyCandle] = {}

        self.last_exit_price = None



        # FEATURES

        self.natr_d0: float | None = None
        self.natr_d1: float | None = None

        self.d0_regime_according_to_natr: RegimeAccordingToNATR | None = None
        self.d1_regime_according_to_natr: RegimeAccordingToNATR | None = None
        self.d0_regime_according_to_natr_probablisticly: float | None = None
        self.d1_regime_according_to_natr_probablisticly: float | None = None

        self.my_natr = self.natr(
            self.current_ticker,
            period=algorithm_parameters.atr_days_period,
            resolution=Resolution.DAILY,
        )

        self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far: list[
            datetime
        ] = []
        self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far: list[
            datetime
        ] = []

        self.intraday_slope_percentile_average_of_last_2_days: float = 0
        self.intraday_slope_percentile_of_last_day: float = 0

        self.intraday_trend_direction_of_last_day: (
            IntradayTrendClassification | None
        ) = None
        self.intraday_trend_directions_of_last_2_days_are_the_same: bool | None = None

        self.d0_trend_value: float | None = None
        self.d1_trend_value: float | None = None
        self.d0_trend_lh_value: float | None = None
        self.d1_trend_lh_value: float | None = None
        self.d2_trend_lh_value: float | None = None
        self.d0_gap_magnitude_value: float | None = None
        self.d1_gap_magnitude_value: float | None = None
        self.d0_gap_magnitude_classification: OvernightGapClassification | None = None
        self.d1_gap_magnitude_classification: OvernightGapClassification | None = None
        self.two_day_average_gap_value: float | None = None

        self.all_values_for_rolling_std: list[float] = []

        self.rolling_std_values: list[float | None] = [None] * len(
            algorithm_parameters.windows_for_rolling_std
        )

        self.opening_range_of_5_min = OpeningRangeFactory(
            name="Opening Range of 5min", opening_ranges_window_minutes=5
        )
        self.opening_range_of_10_min = OpeningRangeFactory(
            name="Opening Range of 10min", opening_ranges_window_minutes=10
        )
        self.opening_range_of_15_min = OpeningRangeFactory(
            name="Opening Range of 15min", opening_ranges_window_minutes=15
        )

        self.intraday_trend_by_date: dict[date, IntradayTrend] = {}
        self.intraday_slopes_last_days: list[float] = []

        self.in_exit_mode: bool = False
        self.in_exit_mode_of_buy: bool = False
        self.in_exit_mode_of_sell: bool = False

        self.risk_management_stop_loss_price_diference: float = 0

        self.realized_returns_by_date = {}  # type: dict[date, list[float]]
        self.should_stop_trading_for_today = False

        # END FEATURES

        self.is_current_market_open = False
        self.market_just_closed = False
        self.is_first_data_after_market_open = False

        self.days_when_market_was_open: list[date] = []

        self.exit_area_when_selling: dict[datetime, float] = {}
        self.exit_area_when_buying: dict[datetime, float] = {}
        self.current_difference_until_stop_loss: float | None = None

        self.current_consecutive_closing_candles_on_exit_area: int = 0

        self.traded_a_buy_today: bool = False
        self.traded_a_sell_today: bool = False

        self.trade_entry_moment = None

        self.first_candle_of_the_day: MyCandle | None = None

        self.take_profit_area: TakeProfitArea | None = None

        self.all_1_minute_candles_by_time: dict[datetime, MyCandle] = {}

        self.init_signals_and_trade_deciders()

    def my_debug(self, arg) -> None:
        if self.live_mode:
            if not self.is_warming_up:
                self.debug(arg)
        else:
            self.debug(arg)

    def memory_cleanup(self):
        self.all_daily_data.clear()
        self.first_candle_of_the_day = None
        self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far.clear()
        self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far.clear()
        self.intraday_slope_percentile_average_of_last_2_days = 0
        self.intraday_slope_percentile_of_last_day = 0

    def init_signals_and_trade_deciders(self) -> None:
        # this is meant to run at init and at the beginning of a market day (to refresh the data)

        self.first_candle_of_the_day = None
        self.take_profit_area = None

        ## FEATURES

        self.consolidation_box_factory_of_5_minutes_op = ConsolidationBoxFactory(
            name="Consolidation Boxes of Opening Range 5min",
            consolidation_box_min_minutes=algorithm_parameters.consolidation_box_min_minutes,
            consolidation_boxes_max_overlap_threshold=algorithm_parameters.consolidation_boxes_max_overlap_threshold,
            consolidation_box_post_minimum_minutes_tolerance_coeficient=algorithm_parameters.consolidation_box_post_minimum_minutes_tolerance_coeficient,
            accumulated_candle_length_for_consolidation_box_exit_evaluation=algorithm_parameters.accumulated_candle_length_for_consolidation_box_exit_evaluation,
            minutes_to_consider_a_hod_or_lod_after_close=algorithm_parameters.cons_box_5_min_minutes_to_consider_a_hod_or_lod_after_close,
            minutes_to_consider_a_hod_or_lod_before_close=algorithm_parameters.cons_box_5_min_minutes_to_consider_a_hod_or_lod_before_close,
            allow_consolidation_boxes_to_exist_on_opening_range_area=algorithm_parameters.allow_consolidation_boxes_to_exist_on_opening_range_area,
        )
        self.consolidation_box_factory_of_10_minutes_op = ConsolidationBoxFactory(
            name="Consolidation Boxes of Opening Range 10min",
            consolidation_box_min_minutes=algorithm_parameters.consolidation_box_min_minutes,
            consolidation_boxes_max_overlap_threshold=algorithm_parameters.consolidation_boxes_max_overlap_threshold,
            consolidation_box_post_minimum_minutes_tolerance_coeficient=algorithm_parameters.consolidation_box_post_minimum_minutes_tolerance_coeficient,
            accumulated_candle_length_for_consolidation_box_exit_evaluation=algorithm_parameters.accumulated_candle_length_for_consolidation_box_exit_evaluation,
            minutes_to_consider_a_hod_or_lod_after_close=algorithm_parameters.cons_box_10_min_minutes_to_consider_a_hod_or_lod_after_close,
            minutes_to_consider_a_hod_or_lod_before_close=algorithm_parameters.cons_box_10_min_minutes_to_consider_a_hod_or_lod_before_close,
            allow_consolidation_boxes_to_exist_on_opening_range_area=algorithm_parameters.allow_consolidation_boxes_to_exist_on_opening_range_area,
        )
        self.consolidation_box_factory_of_15_minutes_op = ConsolidationBoxFactory(
            name="Consolidation Boxes of Opening Range 15min",
            consolidation_box_min_minutes=algorithm_parameters.consolidation_box_min_minutes,
            consolidation_boxes_max_overlap_threshold=algorithm_parameters.consolidation_boxes_max_overlap_threshold,
            consolidation_box_post_minimum_minutes_tolerance_coeficient=algorithm_parameters.consolidation_box_post_minimum_minutes_tolerance_coeficient,
            accumulated_candle_length_for_consolidation_box_exit_evaluation=algorithm_parameters.accumulated_candle_length_for_consolidation_box_exit_evaluation,
            minutes_to_consider_a_hod_or_lod_after_close=algorithm_parameters.cons_box_15_min_minutes_to_consider_a_hod_or_lod_after_close,
            minutes_to_consider_a_hod_or_lod_before_close=algorithm_parameters.cons_box_15_min_minutes_to_consider_a_hod_or_lod_before_close,
            allow_consolidation_boxes_to_exist_on_opening_range_area=algorithm_parameters.allow_consolidation_boxes_to_exist_on_opening_range_area,
        )

        self.consolidation_box_factories = [
            self.consolidation_box_factory_of_5_minutes_op,
            self.consolidation_box_factory_of_10_minutes_op,
            self.consolidation_box_factory_of_15_minutes_op,
        ]

        ## END FEATURES

        # TRADE DECIDERS

        # self.my_simple_trade_decider = SimpleTradeDecider(
        #     name="My Simple 9:45 away from Opening Range Trade Decider",
        #     quantity_to_trade=1,
        # )

        # self.consolidation_box_breaker_decider = ConsolidationBoxBreakerDecider(
        #     name="Consolidation Box Breaker Decider",
        #     quantity_to_trade=1,
        # )

        self.high_volatility_box_breaker_trade_decider = HighVolatilityBoxBreakingTradeDecider(
            name="High Volatility Box Breaking Trade Decider",
            quantity_to_trade=100,
            requires_15_min_box_confirmation=algorithm_parameters.require_15_min_box_confirmation_on_the_trade_decider,
            ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation=algorithm_parameters.ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation,
        )

        # END TRADE DECIDERS

        # SIGNALS

        self.are_last_x_candles_closing_inside_the_exit_area_signal = AreTheLastXCandlesClosingInsideTheExitAreaSignal(
            name="are_last_x_candles_closing_inside_the_exit_area_signal",
            min_number_of_consecutive_candles=algorithm_parameters.min_consecutive_closing_candles_on_exit_area_to_perform_exit,
        )



        self.is_d1_trend_indicating_volatility_signal = (
            IsD1TrendIndicatingVolatilitySignal(
                name="is_d1_trend_indicating_volatility_signal"
            )
        )

        self.is_d1_trend_lh_and_percentile_indicating_volatility_signal = (
            IsD1TrendLHAndPercentileIndicatingVolatilitySignal(
                name="is_d1_trend_lh_and_percentile_indicating_volatility_signal"
            )
        )

        self.has_x_closing_hlods_until_limit_signal = HasXorMoreClosingHLODsUntilYSignal(
            name="has_x_closing_hlods_until_limit_signal",
            hour_of_limit=algorithm_parameters.has_x_closing_hlods_until_limit_signal_hour_of_limit,
            minute_of_limit=algorithm_parameters.has_x_closing_hlods_until_limit_signal_minute_of_limit,
            enough_balance_for_a_signal=algorithm_parameters.has_x_closing_hlods_until_limit_signal_enough_balance_for_a_signal,
        )

        self.is_last_intraday_trend_directions_upward_signal = (
            IsLastDayIntradayTrendDirectionUpwardSignal(
                name="is_last_intraday_trend_directions_upward_signal",
            )
        )

        self.is_last_intraday_trend_directions_downward_signal = (
            IsLastDayIntradayTrendDirectionDownwardSignal(
                name="is_last_intraday_trend_directions_downward_signal",
            )
        )

        self.are_last_2_intraday_trend_directions_the_same_signal = (
            AreLast2IntradayTrendDirectionsTheSameSignal(
                name="are_last_2_intraday_trend_directions_the_same_signal",
            )
        )

        self.average_slope_1_day_percentile_bigger_than_95_threshold_signal = (
            AverageSlopePercentileIsBiggerThanThresholdSignal(
                name="average_slope_1_day_percentile_bigger_than_95_threshold_signal",
                threshold=95,
            )
        )

        self.average_slope_2_days_percentile_bigger_than_75_threshold_signal = (
            AverageSlopePercentileIsBiggerThanThresholdSignal(
                name="average_slope_2_days_percentile_bigger_than_75_threshold_signal",
                threshold=75,
            )
        )

        self.average_slope_2_days_percentile_bigger_than_90_threshold_signal = (
            AverageSlopePercentileIsBiggerThanThresholdSignal(
                name="average_slope_2_days_percentile_bigger_than_90_threshold_signal",
                threshold=90,
            )
        )

        self.a_15_min_box_just_closed_up_signal = ABoxJustClosedUpOrDownSignal(
            name="a_15_min_box_just_closed_up_signal",
            looking_for_up=True,
            looking_for_down=False,
        )
        self.a_15_min_box_just_closed_down_signal = ABoxJustClosedUpOrDownSignal(
            name="a_15_min_box_just_closed_down_signal",
            looking_for_up=False,
            looking_for_down=True,
        )

        self.a_box_closed_lower_signal = ABoxJustClosedHigherOrLowerSignal(
            name="a_box_closed_lower_signal",
            looking_for_high=False,
            looking_for_low=True,
        )
        self.a_box_closed_higher_signal = ABoxJustClosedHigherOrLowerSignal(
            name="a_box_closed_higher_signal",
            looking_for_high=True,
            looking_for_low=False,
        )

        self.is_d1_natr_indicating_volatility_signal = IsNATRIndicatingVolatilitySignal(
            name="is_d1_natr_indicating_volatility_signal"
        )
        self.is_d0_overnight_gap_indicating_volatility_signal = (
            IsOvernightGapIndicatingVolatilitySignal(
                name="is_d0_overnight_gap_indicating_volatility_signal"
            )
        )
        self.is_d1_overnight_gap_indicating_volatility_signal = (
            IsOvernightGapIndicatingVolatilitySignal(
                name="is_d1_overnight_gap_indicating_volatility_signal"
            )
        )

        self.is_current_candle_in_the_exit_area_signal = (
            IsCurrentCandleInTheExitAreaSignal(
                name="is_current_candle_in_the_exit_area_signal"
            )
        )
        self.is_945_signal = Is945Signal(name="is_945_signal")
        self.is_1200_signal = Is1200Signal(name="is_1200_signal")
        self.is_current_candle_away_from_opening_range_signal = (
            IsCurrentCandleAwayFromTheOpenValueSignal(
                name="is_current_candle_away_from_opening_range_signal", threshold=0.002
            )
        )
        
        # END SIGNALS

    def add_open_market_day(self, day: date) -> None:
        if (
            not self.days_when_market_was_open
            or day != self.days_when_market_was_open[-1]
        ):
            self.days_when_market_was_open.append(day)

        if (
            len(self.days_when_market_was_open)
            > algorithm_parameters.how_many_days_to_keep_track_of_market_being_open
        ):
            self.days_when_market_was_open = self.days_when_market_was_open[1:]

    def get_previous_open_market_day(self, how_many_before: int = 1) -> date | None:
        if how_many_before > len(self.days_when_market_was_open) - 1:
            return None
        return self.days_when_market_was_open[-1 - how_many_before]

    def update_moments_of_exceeding_closing_candles(self, candle: MyCandle) -> None:
        current_date = self.time.date()
        if current_date not in self.market_high_by_date.keys():
            return

        closing_value: float = candle.close

        if closing_value > self.market_high_by_date[current_date]:
            self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far.append(
                self.time_reasonable
            )

        if closing_value < self.market_low_by_date[current_date]:
            self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far.append(
                self.time_reasonable
            )

    def update_consecutive_closing_candles_on_exit_area(self, candle: MyCandle) -> None:

        current_close = candle.close

        is_in_exit_area: bool = False

        if (
            self.in_exit_mode_of_buy
            and self.time_reasonable in self.exit_area_when_buying.keys()
        ):
            is_in_exit_area = (
                is_in_exit_area
                or current_close <= self.exit_area_when_buying[self.time_reasonable]
            )

        if (
            self.in_exit_mode_of_sell
            and self.time_reasonable in self.exit_area_when_selling.keys()
        ):
            is_in_exit_area = (
                is_in_exit_area
                or current_close >= self.exit_area_when_selling[self.time_reasonable]
            )

        if is_in_exit_area:
            self.current_consecutive_closing_candles_on_exit_area += 1
        else:
            self.current_consecutive_closing_candles_on_exit_area = 0

    def update_current_day_high_and_low(self, candle: MyCandle) -> None:
        current_high = candle.high
        current_low = candle.low
        current_date = self.time.date()
        if candle.price:
            if current_date not in self.market_high_by_date.keys():
                self.market_high_by_date[current_date] = current_high
                self.market_high_moment_by_date[current_date] = self.time_reasonable
                self.market_low_by_date[current_date] = current_low
                self.market_low_moment_by_date[current_date] = self.time_reasonable
            else:
                if self.market_high_by_date[current_date] < current_high:
                    self.market_high_by_date[current_date] = current_high
                    self.market_high_moment_by_date[current_date] = self.time_reasonable
                if self.market_low_by_date[current_date] > current_low:
                    self.market_low_by_date[current_date] = current_low
                    self.market_low_moment_by_date[current_date] = self.time_reasonable

    def deal_with_first_data_after_market_open(self, candle: MyCandle) -> None:
        current_date = self.time.date()
        if candle.price:
            assert current_date not in self.market_open_prices_by_date.keys()
            self.market_open_prices_by_date[current_date] = candle.open

        # check if i have the conditions to calculate the d-0 Gap:
        previous_day_date = self.get_previous_open_market_day(how_many_before=1)

        if (
            previous_day_date
            and current_date in self.market_open_prices_by_date.keys()
            and previous_day_date in self.market_close_prices_by_date.keys()
        ):
            self.d0_gap_magnitude_value = dx_gap_magnitude_feature(
                open_price_of_current_day=self.market_open_prices_by_date[current_date],
                close_price_of_previous_day=self.market_close_prices_by_date[
                    previous_day_date
                ],
            )

            self.d0_gap_magnitude_classification = classify_dx_gap_magnitude(
                value=self.d0_gap_magnitude_value, d=0
            )

        # check if i have the conditions to calculate the d-1 Gap:
        day_before_yesterday = self.get_previous_open_market_day(how_many_before=2)
        if (
            previous_day_date
            and day_before_yesterday
            and previous_day_date in self.market_open_prices_by_date.keys()
            and day_before_yesterday in self.market_close_prices_by_date.keys()
        ):
            self.d1_gap_magnitude_value = dx_gap_magnitude_feature(
                open_price_of_current_day=self.market_open_prices_by_date[
                    previous_day_date
                ],
                close_price_of_previous_day=self.market_close_prices_by_date[
                    day_before_yesterday
                ],
            )
            self.d1_gap_magnitude_classification = classify_dx_gap_magnitude(
                value=self.d1_gap_magnitude_value, d=1
            )

        # check if i have the conditions to calculate the Gap Average:
        if (
            previous_day_date
            and day_before_yesterday
            and current_date in self.market_open_prices_by_date.keys()
            and previous_day_date in self.market_close_prices_by_date.keys()
            and previous_day_date in self.market_open_prices_by_date.keys()
            and day_before_yesterday in self.market_close_prices_by_date.keys()
        ):
            self.two_day_average_gap_value = gap_magnitude_2_day_average_feature(
                open_price_of_current_day=self.market_open_prices_by_date[current_date],
                close_price_of_previous_day=self.market_close_prices_by_date[
                    previous_day_date
                ],
                open_price_of_previous_day=self.market_open_prices_by_date[
                    previous_day_date
                ],
                close_price_of_day_before_yesterday=self.market_close_prices_by_date[
                    day_before_yesterday
                ],
            )

        if self.intraday_slopes_last_days:

            risk_management_stop_loss_price_diference_slope_value = np.percentile(
                [abs(x) for x in self.intraday_slopes_last_days],
                algorithm_parameters.percentile_of_slope_for_risk_management,
            )

            virtual_closing_price = get_closing_price_from_knowing_trend_and_open_price(
                candle.open,
                risk_management_stop_loss_price_diference_slope_value,
            )

            self.risk_management_stop_loss_price_diference = (
                virtual_closing_price - candle.open
            )

    def market_open_callback(self) -> None:
        self.debug(f"{self.time.date()} // {self.portfolio.total_portfolio_value}")
        self.market_was_ever_open = True
        self.should_stop_trading_for_today = False  # Reset flag at start of day
        if self.time.date() in self.days_to_ignore:
            return

        self.in_exit_mode = False
        self.current_consecutive_closing_candles_on_exit_area = 0
        self.in_exit_mode_of_buy = False
        self.in_exit_mode_of_sell = False
        self.traded_a_buy_today = False
        self.traded_a_sell_today = False

        self.exit_area_when_selling = {}
        self.value_with_most_profit_when_selling: float | None = None
        self.exit_area_when_buying = {}
        self.value_with_most_profit_when_buying: float | None = None
        self.current_difference_until_stop_loss = None

        self.natr_d1 = self.natr_d0
        self.natr_d0 = None

        self.d1_regime_according_to_natr = self.d0_regime_according_to_natr
        self.d0_regime_according_to_natr = None

        self.d1_regime_according_to_natr_probablisticly = (
            self.d0_regime_according_to_natr_probablisticly
        )
        self.d0_regime_according_to_natr_probablisticly = None

        self.is_current_market_open = True
        self.last_market_open_moment = datetime(
            year=self.time.year,
            month=self.time.month,
            day=self.time.day,
            hour=self.time.hour,
            minute=self.time.minute,
            second=0,
        )

        self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far = (
            []
        )
        self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far = (
            []
        )
        self.erase_all_consolidation_boxes()
        self.init_signals_and_trade_deciders()

        self.add_open_market_day(self.time.date())

        self.is_first_data_after_market_open = True

        self.all_daily_data = {}

    def market_close_callback(self) -> None:

        if not self.market_was_ever_open:
            return

        if self.time.date() in self.days_to_ignore:
            return

        self.is_current_market_open = False
        self.market_just_closed = True

    def on_last_data_after_market_close(self, candle: MyCandle) -> None:
        current_date = self.time.date()

        assert current_date not in self.market_close_prices_by_date.keys()
        self.market_close_prices_by_date[current_date] = candle.close

        # LOG THE DAYS HIGH AND LOW, open and close
        if current_date in self.market_open_prices_by_date.keys():
            self.log_feature("Open", self.market_open_prices_by_date[current_date])
        if current_date in self.market_close_prices_by_date.keys():
            self.log_feature("Close", self.market_close_prices_by_date[current_date])
        if current_date in self.market_high_by_date.keys():
            self.log_feature("High", self.market_high_by_date[current_date])
        if current_date in self.market_low_by_date.keys():
            self.log_feature("Low", self.market_low_by_date[current_date])

        # log the moments where candles closing values exceeded high or low of the day
        feature_name = "Moments of Candles Closing on new highs"
        self.log_feature(
            feature_name,
            self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
            is_datetime_list=True,
        )
        feature_name = "Moments of Candles Closing on new lows"
        self.log_feature(
            feature_name,
            self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
            is_datetime_list=True,
        )

        # log if possible
        if self.d0_gap_magnitude_value is not None:
            feature_name = "D-0 Gap"
            self.log_feature(feature_name, self.d0_gap_magnitude_value)
            feature_name = "D-0 Gap Classification"
            self.log_feature(feature_name, self.d0_gap_magnitude_classification.value)
        if self.d1_gap_magnitude_value is not None:
            feature_name = "D-1 Gap"
            self.log_feature(
                feature_name=feature_name, feature_value=self.d1_gap_magnitude_value
            )
            feature_name = "D-1 Gap Classification"
            self.log_feature(feature_name, self.d1_gap_magnitude_classification.value)
        if self.two_day_average_gap_value is not None:
            feature_name = "2 Day Gap Avg"
            self.log_feature(feature_name, self.two_day_average_gap_value)

        for op_range in [
            (self.opening_range_of_5_min),
            (self.opening_range_of_10_min),
            (self.opening_range_of_15_min),
        ]:
            if current_date in op_range.opening_ranges_by_date.keys():
                cur_opening_range = op_range.opening_ranges_by_date[current_date]
                for keyword, cur_val in (
                    ("Low", cur_opening_range.low),
                    ("High", cur_opening_range.high),
                ):
                    feature_name = f"{op_range.name} {keyword} "
                    self.log_feature(feature_name, cur_val)

        # check if i have the conditions to calculate the d-0 Trend:
        if (
            current_date in self.market_close_prices_by_date.keys()
            and current_date in self.market_open_prices_by_date.keys()
        ):
            self.d0_trend_value = dx_trend_feature(
                open_price=self.market_open_prices_by_date[current_date],
                close_price=self.market_close_prices_by_date[current_date],
            )
            feature_name = "D-0 Trend"
            self.log_feature(feature_name, self.d0_trend_value)

        # check if i have the conditions to calculate the d-1 Trend:
        previous_day_date = self.get_previous_open_market_day(how_many_before=1)
        day_before_prev_day_date = self.get_previous_open_market_day(how_many_before=2)
        if (
            previous_day_date
            and previous_day_date in self.market_close_prices_by_date.keys()
            and previous_day_date in self.market_open_prices_by_date.keys()
        ):
            self.d1_trend_value = dx_trend_feature(
                open_price=self.market_open_prices_by_date[previous_day_date],
                close_price=self.market_close_prices_by_date[previous_day_date],
            )
            feature_name = "D-1 Trend"
            self.log_feature(feature_name, self.d1_trend_value)

        self.d0_trend_lh_value = dx_trend_lh_feature(
            lowest_price=self.market_low_by_date[self.time.date()],
            highest_price=self.market_high_by_date[self.time.date()],
            close_price=self.market_close_prices_by_date[self.time.date()],
        )
        feature_name = "D-0 Trend LH"
        self.log_feature(feature_name, self.d0_trend_lh_value)

        # check if i have the conditions to calculate the d-1 LH Trend:
        if (
            previous_day_date
            and previous_day_date in self.market_close_prices_by_date.keys()
            and previous_day_date in self.market_high_by_date.keys()
            and previous_day_date in self.market_low_by_date.keys()
        ):
            d1_trend_lh_value_of_the_prev_day = dx_trend_lh_feature(
                lowest_price=self.market_low_by_date[previous_day_date],
                highest_price=self.market_high_by_date[previous_day_date],
                close_price=self.market_close_prices_by_date[previous_day_date],
            )

            if self.d1_trend_lh_value is not None:
                self.d2_trend_lh_value = self.d1_trend_lh_value

            self.d1_trend_lh_value = d1_trend_lh_value_of_the_prev_day

            feature_name = "D-1 Trend LH"
            self.log_feature(feature_name, self.d1_trend_lh_value)

        if self.d2_trend_lh_value is not None:
            feature_name = "D-2 Trend LH"
            self.log_feature(feature_name, self.d2_trend_lh_value)

        if current_date in self.market_high_moment_by_date.keys():
            self.log_feature(
                "Current Day Market High moment",
                self.market_high_moment_by_date[current_date],
            )

        if current_date in self.market_low_moment_by_date.keys():
            self.log_feature(
                "Current Day Market Low moment",
                self.market_low_moment_by_date[current_date],
            )

        if current_date in self.market_high_by_date.keys():
            self.log_feature(
                "Current Day Market High value",
                self.market_high_by_date[current_date],
            )

        if current_date in self.market_low_by_date.keys():
            self.log_feature(
                "Current Day Market Low value",
                self.market_low_by_date[current_date],
            )

        # log intraday trends
        assert current_date in self.intraday_trend_by_date.keys()
        intraday_trend = self.intraday_trend_by_date[current_date]

        self.log_feature(
            "Intraday Slope Percentile Threshold",
            algorithm_parameters.intraday_trend_classification_percentile,
        )

        self.log_feature(
            "Minimum Consecutive Closing Candles on Exit Area to perform Exit",
            algorithm_parameters.min_consecutive_closing_candles_on_exit_area_to_perform_exit,
        )

        self.log_feature(
            "D-0 Intraday Trend Classification", intraday_trend.classification.value
        )
        self.log_feature("D-0 Intraday Trend Slope", intraday_trend.slope)
        self.log_feature(
            "D-0 Intraday Trend Slope respective Percentile", intraday_trend.percentile
        )

        if previous_day_date:
            assert previous_day_date in self.intraday_trend_by_date.keys()
            intraday_trend = self.intraday_trend_by_date[previous_day_date]
            self.log_feature(
                "D-1 Intraday Trend Classification", intraday_trend.classification.value
            )
            self.log_feature("D-1 Intraday Trend Slope", intraday_trend.slope)
            self.log_feature(
                "D-1 Intraday Trend Slope respective Percentile",
                intraday_trend.percentile,
            )

        if day_before_prev_day_date:
            assert day_before_prev_day_date in self.intraday_trend_by_date.keys()
            intraday_trend = self.intraday_trend_by_date[day_before_prev_day_date]
            self.log_feature(
                "D-2 Intraday Trend Classification", intraday_trend.classification.value
            )
            self.log_feature("D-2 Intraday Trend Slope", intraday_trend.slope)
            self.log_feature(
                "D-2 Intraday Trend Slope respective Percentile",
                intraday_trend.percentile,
            )

        self.log_feature(
            "Intraday Percentile Average of Last 2 days",
            self.intraday_slope_percentile_average_of_last_2_days,
        )

        self.log_feature(
            "Exit Area of Buy",
            (
                self.convert_dict_of_datetime_key_to_strf_version(
                    self.exit_area_when_buying
                )
                if self.traded_a_buy_today
                else {}
            ),
        )
        self.log_feature(
            "Exit Area of Sell",
            (
                self.convert_dict_of_datetime_key_to_strf_version(
                    self.exit_area_when_selling
                )
                if self.traded_a_sell_today
                else {}
            ),
        )

        if self.take_profit_area:
            self.Debug(f"TPA populated? {len(self.take_profit_area.take_profit_area_at_moment)} entries | traded_long={self.take_profit_area.traded_long}")

        self.log_feature(
            "Take Profit Area of Buy",
            (
                self.convert_dict_of_datetime_key_to_strf_version(
                    self.take_profit_area.take_profit_area_at_moment
                )
                if self.take_profit_area and self.take_profit_area.traded_long
                else {}
            ),
        )

        self.log_feature(
            "Take Profit Area of Sell",
            (
                self.convert_dict_of_datetime_key_to_strf_version(
                    self.take_profit_area.take_profit_area_at_moment
                )
                if self.take_profit_area and not self.take_profit_area.traded_long
                else {}
            ),
        )

        self.log_feature(
            "Risk Management: Percentile of Slope",
            algorithm_parameters.percentile_of_slope_for_risk_management,
        )

        self.log_feature(
            "Risk Management: Stop Loss price diference",
            self.risk_management_stop_loss_price_diference,
        )

    T = TypeVar("T")

    def convert_dict_of_datetime_key_to_strf_version(
        self, datetime_dict: dict[datetime, T]
    ) -> dict[str, T]:
        final_dict = {}
        for x in datetime_dict:
            final_dict[x.strftime(STRFTIME_FORMAT)] = datetime_dict[x]
        return final_dict

    def update_rolling_std(self) -> None:

        new_value_for_rolling_std = self.d0_trend_value

        # add the new value
        self.all_values_for_rolling_std.append(new_value_for_rolling_std)

        # check if the list of values is already too big
        max_window_size = max(algorithm_parameters.windows_for_rolling_std)
        if len(self.all_values_for_rolling_std) > max_window_size:
            self.all_values_for_rolling_std = self.all_values_for_rolling_std[
                len(self.all_values_for_rolling_std) - max_window_size :
            ]
        assert len(self.all_values_for_rolling_std) <= max_window_size

        for current_rolling_std_index, current_rolling_std_window in enumerate(
            algorithm_parameters.windows_for_rolling_std
        ):

            if len(self.all_values_for_rolling_std) >= current_rolling_std_window:
                new_result = rolling_standard_deviation_feature(
                    self.all_values_for_rolling_std,
                    window_size=current_rolling_std_window,
                )

                self.rolling_std_values[current_rolling_std_index] = new_result

                feature_name = f"Daily Rolling STD (of D-trends) (Window size {current_rolling_std_window})"
                self.log_feature(
                    feature_name, self.rolling_std_values[current_rolling_std_index]
                )

    def log_feature(
        self, feature_name: str, feature_value: any, is_datetime_list: bool = False
    ) -> None:
        if self.enable_logs is False:
            return
        final_value = (
            round(feature_value, 7) if type(feature_value) in [float] else feature_value
        )

        if is_datetime_list:
            final_value = [x.strftime(STRFTIME_FORMAT) for x in feature_value]

        if type(final_value) == datetime:
            final_value = final_value.strftime(STRFTIME_FORMAT)

        self.my_debug(
            f"FEATURE : name: {feature_name} // moment: {self.time.date()}__{self.time.time()}  // value: {final_value} "
        )

    def log_current_candle(self, current_candle: MyCandle, current_symbol: Symbol):
        if self.enable_logs is False:
            return
        self.my_debug(
            f"CANDLE : Ticker: {current_symbol.to_string()} // moment: {self.time.date()}__{self.time.time()} "
            f"// price: {round(current_candle.price,7)} "
            f"// open: {round(current_candle.open,7)} "
            f"// high: {round(current_candle.high,7)} "
            f"// low: {round(current_candle.low,7)} "
            f"// close: {round(current_candle.close,7)} "
            f"// volume: {round(current_candle.volume,7)} "
        )

    def log_trade_exit(
        self, trade_name: str, trade_identifier: str, price_on_exit: str
    ) -> None:
        if self.enable_logs is False:
            return
        debug_message = (
            f"TRADE EXIT /// "
            f"exit_moment: {self.time.date()}__{self.time.time()}  // "
            f"name: {trade_name} // "
            f"identifier: {trade_identifier} // "
            f"price_on_exit: {price_on_exit}  // "
        )
        #self.notify.sms(phone_number="+18313315114", message=f"Notify Exit: " + debug_message)
        #self.notify.sms(phone_number="+12162245775", message=f"Notify Exit: " + debug_message)
        self.my_debug(debug_message)

    def liquidate_position(self, reason: str):
        quantity = self.Portfolio[self.current_ticker].Quantity
        if quantity != 0:
            self.Debug(f"LIQUIDATING {quantity} shares at {self.Time} — Reason: {reason}")
            self.MarketOrder(self.current_ticker, -quantity)

            current_price = self.Securities[self.current_ticker].Price
            self.last_exit_price = current_price  # ✅ Track the exit price for next-trade filtering


            trade_name = self.high_volatility_box_breaker_trade_decider.name or "FORCED_LIQUIDATION"
            trade_identifier = (
                self.high_volatility_box_breaker_trade_decider.trade_decider_message_identifier
                or "FORCED_LIQUIDATION"
            )

            self.log_trade_exit(
                trade_name=trade_name,
                trade_identifier=trade_identifier,
                price_on_exit=str(self.Securities[self.current_ticker].Price)
            )

            self.high_volatility_box_breaker_trade_decider.remove_open_trade()
            self.take_profit_area = None
            self.exit_area = None
            self.trade_entry_moment = None
            self.has_open_trade = False
            self.in_exit_mode = False
            self.in_exit_mode_of_buy = False
            self.in_exit_mode_of_sell = False

    def log_signal(
        self,
        signal_name: str,
        signal_decision: SignalDecisionType,
        signal_certainty: float,
        signal_relevant_data: dict[str, float | str],
    ) -> None:
        if self.enable_logs is False:
            return
        debug_message = (
            f"SIGNAL /// "
            f"moment: {self.time.date()}__{self.time.time()}  // "
            f"name: {signal_name} // "
            f"decision: {signal_decision.value}  // "
            f"certainty: {signal_certainty}  // "
        )

        for relevant_data_key, relevant_data_value in signal_relevant_data.items():
            new_message = f"Relevant -> {relevant_data_key}: {relevant_data_value} // "
            debug_message = debug_message + new_message
        self.my_debug(debug_message)

    def log_trade(
        self,
        trade_name: str,
        trade_identifier: str,
        trade_decision: TradeDecisionType,
        is_opposite: bool,
        is_towards_opening_price: bool | None,
        trade_entry_price: float,
        trade_order_quantity: Optional[float],
        trade_relevant_data: dict[str, float | str],
        trade_feature_data: dict[str, float | str],
        trade_pre_signal_moment: Optional[datetime],
    ) -> None:
        if self.enable_logs is False:
            return

        pre_signal_str = (
            f"{trade_pre_signal_moment.date()}__{trade_pre_signal_moment.time()}"
            if trade_pre_signal_moment
            else None
        )
        debug_message = (
            f"TRADE /// "
            f"moment: {self.time.date()}__{self.time.time()}  // "
            f"name: {trade_name} // "
            f"identifier: {trade_identifier} // "
            f"decision: {trade_decision.value}  // "
            f"entry_price: {trade_entry_price}  // "
            f"order_quantity: {trade_order_quantity or 0} // "
            f"is_opposite: {is_opposite} // "
            f"is_towards_opening_price:  {is_towards_opening_price} // "
            f"pre signal moment: {pre_signal_str}  // "
        )

        #self.notify.sms(phone_number="+18313315114", message=f"Notify Trade: " + debug_message)
        #self.notify.sms(phone_number="+12162245775", message=f"Notify Trade: " + debug_message)

        for relevant_data_key, relevant_data_value in trade_feature_data.items():
            new_message = f"Feature -> {relevant_data_key}: {relevant_data_value} // "
            debug_message = debug_message + new_message

        for relevant_data_key, relevant_data_value in trade_relevant_data.items():
            new_message = f"Relevant -> {relevant_data_key}: {relevant_data_value} // "
            debug_message = debug_message + new_message
        self.my_debug(debug_message)

    def log_consolidation_boxes_feature(self, is_live: bool = False) -> None:

        for cons_box_factory in [
            self.consolidation_box_factory_of_5_minutes_op,
            self.consolidation_box_factory_of_10_minutes_op,
            self.consolidation_box_factory_of_15_minutes_op,
        ]:

            if not is_live:
                assert cons_box_factory.open_consolidation_box is None
            self.log_feature(
                f"{cons_box_factory.name} Min Minutes",
                cons_box_factory.consolidation_box_min_minutes,
            )
            self.log_feature(
                f"{cons_box_factory.name} Max Thickness",
                cons_box_factory.consolidation_box_max_thickness,
            )
            self.log_feature(
                f"{cons_box_factory.name} Thickness Tolerance Coeficient",
                cons_box_factory.consolidation_box_post_minimum_minutes_tolerance_coeficient,
            )
            self.log_feature(
                f"{cons_box_factory.name} Max Overlap Threshold",
                cons_box_factory.consolidation_boxes_max_overlap_threshold,
            )
            self.log_feature(
                f"{cons_box_factory.name} #",
                len(cons_box_factory.closed_consolidation_boxes),
            )
            self.log_feature(
                f"{cons_box_factory.name} Values",
                [
                    x.model_dump_json()
                    for x in cons_box_factory.closed_consolidation_boxes
                ],
            )

            self.log_feature(
                f"{cons_box_factory.name} Open Box",
                (
                    cons_box_factory.open_consolidation_box.model_dump_json()
                    if cons_box_factory.open_consolidation_box
                    else cons_box_factory.open_consolidation_box
                ),
            )

    def execute_signals(self, candle: MyCandle) -> None:
        current_date = self.time.date()
        current_time = self.time_reasonable
        current_close_price = candle.close
        current_day_open_price = self.market_open_prices_by_date[current_date]
        previous_day_date = self.get_previous_open_market_day(how_many_before=1)

        ####

        signal_name = self.are_last_x_candles_closing_inside_the_exit_area_signal.name
        signal_decision = self.are_last_x_candles_closing_inside_the_exit_area_signal.decide(
            current_consecutive_closing_candles_on_exit_area=self.current_consecutive_closing_candles_on_exit_area
        )

        if (
            self.are_last_x_candles_closing_inside_the_exit_area_signal.most_recent_decision
            is None
            or self.are_last_x_candles_closing_inside_the_exit_area_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.are_last_x_candles_closing_inside_the_exit_area_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        

        ####

        ####

        signal_name = self.is_d1_trend_indicating_volatility_signal.name
        signal_decision = self.is_d1_trend_indicating_volatility_signal.decide(
            d1_intraday_trend=(
                self.intraday_trend_by_date[previous_day_date]
                if previous_day_date
                else None
            )
        )

        if (
            self.is_d1_trend_indicating_volatility_signal.most_recent_decision is None
            or self.is_d1_trend_indicating_volatility_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_d1_trend_indicating_volatility_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = (
            self.is_d1_trend_lh_and_percentile_indicating_volatility_signal.name
        )
        signal_decision = (
            self.is_d1_trend_lh_and_percentile_indicating_volatility_signal.decide(
                d1_trend_lh=self.d0_trend_lh_value
                or 0,  # TODO: fix this d0 lh instead of d1 lh
                d1_trend_percentile=(
                    self.intraday_trend_by_date[previous_day_date].percentile
                    if previous_day_date
                    else None
                ),
            )
        )

        if (
            self.is_d1_trend_lh_and_percentile_indicating_volatility_signal.most_recent_decision
            is None
            or self.is_d1_trend_lh_and_percentile_indicating_volatility_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_d1_trend_lh_and_percentile_indicating_volatility_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = self.has_x_closing_hlods_until_limit_signal.name
        signal_decision = self.has_x_closing_hlods_until_limit_signal.decide(
            moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far=self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
            moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far=self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
            current_time=self.time,
            current_price=candle.close,
            opening_range_factories_to_consider=[
                self.get_opening_range_mainly_used(x)
                for x in algorithm_parameters.minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal
            ],
            percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal=algorithm_parameters.percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal
        )

        if (
            self.has_x_closing_hlods_until_limit_signal.most_recent_decision is None
            or self.has_x_closing_hlods_until_limit_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.has_x_closing_hlods_until_limit_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = self.is_last_intraday_trend_directions_downward_signal.name
        signal_decision = self.is_last_intraday_trend_directions_downward_signal.decide(
            intraday_trend_classification=self.intraday_trend_direction_of_last_day
        )

        if (
            self.is_last_intraday_trend_directions_downward_signal.most_recent_decision
            is None
            or self.is_last_intraday_trend_directions_downward_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_last_intraday_trend_directions_downward_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = self.is_last_intraday_trend_directions_upward_signal.name
        signal_decision = self.is_last_intraday_trend_directions_upward_signal.decide(
            intraday_trend_classification=self.intraday_trend_direction_of_last_day
        )

        if (
            self.is_last_intraday_trend_directions_upward_signal.most_recent_decision
            is None
            or self.is_last_intraday_trend_directions_upward_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_last_intraday_trend_directions_upward_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = self.are_last_2_intraday_trend_directions_the_same_signal.name
        signal_decision = (
            self.are_last_2_intraday_trend_directions_the_same_signal.decide(
                are_the_same=self.intraday_trend_directions_of_last_2_days_are_the_same
            )
        )

        if (
            self.are_last_2_intraday_trend_directions_the_same_signal.most_recent_decision
            is None
            or self.are_last_2_intraday_trend_directions_the_same_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.are_last_2_intraday_trend_directions_the_same_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = (
            self.average_slope_1_day_percentile_bigger_than_95_threshold_signal.name
        )
        signal_decision = (
            self.average_slope_1_day_percentile_bigger_than_95_threshold_signal.decide(
                average_percentile=self.intraday_slope_percentile_of_last_day
            )
        )

        if (
            self.average_slope_1_day_percentile_bigger_than_95_threshold_signal.most_recent_decision
            is None
            or self.average_slope_1_day_percentile_bigger_than_95_threshold_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.average_slope_1_day_percentile_bigger_than_95_threshold_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = (
            self.average_slope_2_days_percentile_bigger_than_75_threshold_signal.name
        )
        signal_decision = (
            self.average_slope_2_days_percentile_bigger_than_75_threshold_signal.decide(
                average_percentile=self.intraday_slope_percentile_average_of_last_2_days
            )
        )

        if (
            self.average_slope_2_days_percentile_bigger_than_75_threshold_signal.most_recent_decision
            is None
            or self.average_slope_2_days_percentile_bigger_than_75_threshold_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.average_slope_2_days_percentile_bigger_than_75_threshold_signal.most_recent_decision = (
            signal_decision
        )

        ####

        ####

        signal_name = (
            self.average_slope_2_days_percentile_bigger_than_90_threshold_signal.name
        )
        signal_decision = (
            self.average_slope_2_days_percentile_bigger_than_90_threshold_signal.decide(
                average_percentile=self.intraday_slope_percentile_average_of_last_2_days
            )
        )

        if (
            self.average_slope_2_days_percentile_bigger_than_90_threshold_signal.most_recent_decision
            is None
            or self.average_slope_2_days_percentile_bigger_than_90_threshold_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.average_slope_2_days_percentile_bigger_than_90_threshold_signal.most_recent_decision = (
            signal_decision
        )

        ####
        ####

        signal_name = self.a_15_min_box_just_closed_down_signal.name
        signal_decision = self.a_15_min_box_just_closed_down_signal.decide(
            current_datetime=self.time_reasonable,
            consolidation_box_factories=[
                self.consolidation_box_factory_of_15_minutes_op,
            ],
        )

        if (
            self.a_15_min_box_just_closed_down_signal.most_recent_decision is None
            or self.a_15_min_box_just_closed_down_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )
        

        self.a_15_min_box_just_closed_down_signal.most_recent_decision = signal_decision



        signal_name = self.a_15_min_box_just_closed_up_signal.name
        signal_decision = self.a_15_min_box_just_closed_up_signal.decide(
            current_datetime=self.time_reasonable,
            consolidation_box_factories=[
                self.consolidation_box_factory_of_15_minutes_op,
            ],
        )

        if (
            self.a_15_min_box_just_closed_up_signal.most_recent_decision is None
            or self.a_15_min_box_just_closed_up_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        

        self.a_15_min_box_just_closed_up_signal.most_recent_decision = signal_decision

        signal_name = self.a_box_closed_lower_signal.name
        signal_decision = self.a_box_closed_lower_signal.decide(
            current_datetime=self.time_reasonable,
            consolidation_box_factories=[
                self.consolidation_box_factory_of_5_minutes_op,
                self.consolidation_box_factory_of_10_minutes_op,
                self.consolidation_box_factory_of_15_minutes_op,
            ],
        )

        if (
            self.a_box_closed_lower_signal.most_recent_decision is None
            or self.a_box_closed_lower_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.a_box_closed_lower_signal.most_recent_decision = signal_decision

        ####

        ####

        signal_name = self.a_box_closed_higher_signal.name
        signal_decision = self.a_box_closed_higher_signal.decide(
            current_datetime=self.time_reasonable,
            consolidation_box_factories=[
                self.consolidation_box_factory_of_5_minutes_op,
                self.consolidation_box_factory_of_10_minutes_op,
                self.consolidation_box_factory_of_15_minutes_op,
            ],
        )

        if (
            self.a_box_closed_higher_signal.most_recent_decision is None
            or self.a_box_closed_higher_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.a_box_closed_higher_signal.most_recent_decision = signal_decision

        signal_name = self.is_d1_natr_indicating_volatility_signal.name
        signal_decision = self.is_d1_natr_indicating_volatility_signal.decide(
            natr_classification=self.d1_regime_according_to_natr
        )

        if (
            self.is_d1_natr_indicating_volatility_signal.most_recent_decision is None
            or self.is_d1_natr_indicating_volatility_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_d1_natr_indicating_volatility_signal.most_recent_decision = (
            signal_decision
        )

        ####

        signal_name = self.is_d0_overnight_gap_indicating_volatility_signal.name
        signal_decision = self.is_d0_overnight_gap_indicating_volatility_signal.decide(
            overnight_gap_classification=self.d0_gap_magnitude_classification
        )

        if (
            self.is_d0_overnight_gap_indicating_volatility_signal.most_recent_decision
            is None
            or self.is_d0_overnight_gap_indicating_volatility_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_d0_overnight_gap_indicating_volatility_signal.most_recent_decision = (
            signal_decision
        )

        ####

        signal_name = self.is_d1_overnight_gap_indicating_volatility_signal.name
        signal_decision = self.is_d1_overnight_gap_indicating_volatility_signal.decide(
            overnight_gap_classification=self.d1_gap_magnitude_classification
        )

        if (
            self.is_d1_overnight_gap_indicating_volatility_signal.most_recent_decision
            is None
            or self.is_d1_overnight_gap_indicating_volatility_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_d1_overnight_gap_indicating_volatility_signal.most_recent_decision = (
            signal_decision
        )

        ####

        signal_name = self.is_945_signal.name
        signal_decision = self.is_945_signal.decide(current_time=current_time)

        if (
            self.is_945_signal.most_recent_decision is None
            or self.is_945_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_945_signal.most_recent_decision = signal_decision

        signal_name = self.is_1200_signal.name
        signal_decision = self.is_1200_signal.decide(current_time=current_time)

        if (
            self.is_1200_signal.most_recent_decision is None
            or self.is_1200_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_1200_signal.most_recent_decision = signal_decision

        signal_name = self.is_current_candle_away_from_opening_range_signal.name
        signal_decision = self.is_current_candle_away_from_opening_range_signal.decide(
            current_close_price=current_close_price,
            open_price=current_day_open_price,
        )

        if (
            self.is_current_candle_away_from_opening_range_signal.most_recent_decision
            is None
            or self.is_current_candle_away_from_opening_range_signal.most_recent_decision.decision_type
            != signal_decision.decision_type
        ):
            self.log_signal(
                signal_name=signal_name,
                signal_decision=signal_decision.decision_type,
                signal_certainty=signal_decision.degree_of_certainty,
                signal_relevant_data=signal_decision.relevant_data,
            )

        self.is_current_candle_away_from_opening_range_signal.most_recent_decision = (
            signal_decision
        )

        ## IMPORTANT!!! THIS MUST BE THE LAST SIGNAL TO BE GENERATED

        current_candle = self.all_daily_data.get(self.time_reasonable, None)

        if current_candle is None:
            self.logger.warning(f"[WARNING] No current_candle at {self.time_reasonable} — skipping is_current_candle_in_the_exit_area_signal")
        else:
            signal_name = self.is_current_candle_in_the_exit_area_signal.name
            signal_decision = self.is_current_candle_in_the_exit_area_signal.decide(
                current_candle=current_candle,
                current_moment=self.time_reasonable,
                in_exit_mode_of_buy=self.in_exit_mode_of_buy,
                in_exit_mode_of_sell=self.in_exit_mode_of_sell,
                exit_area_when_buying=self.exit_area_when_buying,
                exit_area_when_selling=self.exit_area_when_selling,
            )

            if (
                self.is_current_candle_in_the_exit_area_signal.most_recent_decision is None
                or self.is_current_candle_in_the_exit_area_signal.most_recent_decision.decision_type != signal_decision.decision_type
            ):
                self.log_signal(
                    signal_name=signal_name,
                    signal_decision=signal_decision.decision_type,
                    signal_certainty=signal_decision.degree_of_certainty,
                    signal_relevant_data=signal_decision.relevant_data,
                )

            self.is_current_candle_in_the_exit_area_signal.most_recent_decision = signal_decision

        ###

    
    def execute_specific_trade_decider(
            self,
            candle: MyCandle,
            trade_decider_function: TradeDecider,
            *args: SignalDecision,
            **kwargs,
        ) -> TradeDecision:

            assert self.risk_management_stop_loss_price_diference

            current_date = self.time.date()
            current_price = candle.close
            assert current_date in self.market_open_prices_by_date.keys()
            
            # Leverage-aware sizing (shares/contracts baseline before sign)
            weight = float(algorithm_parameters.proportion_of_cash_to_trade)
            target_leverage = float(getattr(self, "target_leverage", 1))
            notional = float(self.Portfolio.TotalPortfolioValue) * target_leverage * weight
            self.quantity_to_trade = max(1, int(math.floor(notional / max(1e-9, candle.close))))

            # Optional: if you cap engine leverage elsewhere, keep it; otherwise this is fine.

            # Stop-trading guard based on realized returns
            if hasattr(self, "should_stop_trading_for_today") and self.should_stop_trading_for_today:
                self.debug(f"Skipping trade — a realized return exceeded 2% on {current_date}")
                return TradeDecision(
                    decision_type=TradeDecisionType.DONT_TRADE,
                    order_quantity=self.quantity_to_trade,
                    relevant_data={"note": "Realized return exceeded 2%"},
                    feature_data={},
                    is_opposite=False,
                )

            # ENTRY PATH
            if (
                trade_decider_function.open_trade is None
                and self.is_current_market_open
                and trade_decider_function.can_i_still_trade_today()
            ):
                limit_trade_entering_datetime: datetime | None = (
                    datetime(
                        year=self.time_reasonable.year,
                        month=self.time_reasonable.month,
                        day=self.time_reasonable.day,
                        hour=algorithm_parameters.cant_trade_beyond_this_time.hour,
                        minute=algorithm_parameters.cant_trade_beyond_this_time.minute,
                        second=0,
                    )
                    if algorithm_parameters.cant_trade_beyond_this_time
                    else None
                )

                if (
                    limit_trade_entering_datetime is None
                    or self.time_reasonable <= limit_trade_entering_datetime
                ):
                    trading_decision: TradeDecision = trade_decider_function.decide(
                        *args, **kwargs,
                        consolidation_box_factories=self.consolidation_box_factories,
                        all_1_minute_candles_by_time=self.all_1_minute_candles_by_time,
                        all_daily_data=self.all_daily_data,
                    )
                    trading_decision.pre_signal_moment = trade_decider_function.moment_of_pre_signal
                    trade_decider_function.generate_trade_decider_message_identifier()

                    if trading_decision.decision_type not in [
                        TradeDecisionType.NOT_SURE,
                        TradeDecisionType.DONT_TRADE,
                    ]:

                        # Avoid immediate re-entry near last exit price
                        if self.last_exit_price is not None:
                            if abs(current_price - self.last_exit_price) < 0.01:
                                self.debug("Skipping trade because price is too close to last exit")
                                return TradeDecision(
                                    decision_type=TradeDecisionType.DONT_TRADE,
                                    order_quantity=self.quantity_to_trade,
                                    relevant_data={"note": "Price near last exit"},
                                    feature_data=trading_decision.feature_data,
                                    is_opposite=trading_decision.is_opposite,
                                )

                        # Determine signed order quantity
                        base_qty = int(self.quantity_to_trade)
                        side_sign = -1 if trading_decision.decision_type == TradeDecisionType.SELL else 1
                        order_quantity = side_sign * base_qty

                        self.log_trade(
                            trade_name=trade_decider_function.name,
                            trade_identifier=trade_decider_function.trade_decider_message_identifier,
                            trade_decision=trading_decision.decision_type,
                            is_opposite=trading_decision.is_opposite,
                            is_towards_opening_price=trading_decision.is_towards_opening_price,
                            trade_entry_price=current_price,
                            trade_order_quantity=order_quantity,
                            trade_relevant_data=trading_decision.relevant_data,
                            trade_feature_data=trading_decision.feature_data,
                            trade_pre_signal_moment=trading_decision.pre_signal_moment,
                        )

                        # Reset exit-area state
                        self.exit_area_when_buying = {}
                        self.exit_area_when_selling = {}
                        self.value_with_most_profit_when_buying = None
                        self.value_with_most_profit_when_selling = None
                        self.current_difference_until_stop_loss = None

                        # Place order (futures vs equity) with gentle guards
                        if futures_trading_parameters.is_trading_futures:
                            # Desired contracts (simple notional/price heuristic)
                            desired_contracts = max(1, int(math.floor(notional / max(1e-9, candle.close))))
                            if side_sign < 0:
                                desired_contracts *= -1

                            # Quick conservative cap using available margin if available
                            try:
                                sec = self.Securities[self.future_contract]
                                price = max(1e-9, float(sec.Price))
                                leverage = float(getattr(sec, "Leverage", 1.0))
                                margin_remaining = float(self.Portfolio.MarginRemaining)
                                est_max = int(max(0, math.floor((margin_remaining * leverage) / price)))
                                est_max = int(max(0, math.floor(est_max * 0.98)))  # safety pad
                                if est_max == 0:
                                    self.debug("Not enough margin to place a futures order")
                                    return TradeDecision(
                                        decision_type=TradeDecisionType.DONT_TRADE,
                                        order_quantity=self.quantity_to_trade,
                                        relevant_data={"note": "no_margin_remaining_futures"},
                                        feature_data={},
                                        is_opposite=False,
                                    )
                                send_qty = max(-est_max, min(desired_contracts, est_max))
                            except Exception:
                                # Fallback: at least don't send 0
                                send_qty = desired_contracts

                            if send_qty == 0:
                                self.debug("Blocked: futures qty clipped to 0 by guard")
                                return TradeDecision(
                                    decision_type=TradeDecisionType.DONT_TRADE,
                                    order_quantity=self.quantity_to_trade,
                                    relevant_data={"note": "qty_clipped_to_zero_futures"},
                                    feature_data={},
                                    is_opposite=False,
                                )

                            self.Debug(f"Delaying {ACCOUNT_DELAY_SECONDS} seconds before placing futures order")
                            time_module.sleep(ACCOUNT_DELAY_SECONDS)
                            current_market_order = self.market_order(self.future_contract, int(send_qty))

                        else:
                            # Equity margin guard
                            sec = self.Securities[self.current_ticker]
                            price = max(1e-9, float(sec.Price))
                            leverage = float(getattr(sec, "Leverage", 1.0))
                            margin_remaining = float(self.Portfolio.MarginRemaining)
                            est_max_shares = int(max(0, math.floor((margin_remaining * leverage) / price)))
                            est_max_shares = int(max(0, math.floor(est_max_shares * 0.98)))  # safety pad

                            if est_max_shares == 0:
                                self.debug("Blocked: no margin remaining for equity order")
                                return TradeDecision(
                                    decision_type=TradeDecisionType.DONT_TRADE,
                                    order_quantity=self.quantity_to_trade,
                                    relevant_data={"note": "no_margin_remaining_equity"},
                                    feature_data={},
                                    is_opposite=False,
                                )

                            send_qty = max(-est_max_shares, min(order_quantity, est_max_shares))
                            if send_qty == 0:
                                self.debug("Blocked: desired qty clipped to 0 by margin guard")
                                return TradeDecision(
                                    decision_type=TradeDecisionType.DONT_TRADE,
                                    order_quantity=self.quantity_to_trade,
                                    relevant_data={"note": "qty_clipped_to_zero_equity"},
                                    feature_data={},
                                    is_opposite=False,
                                )

                            self.Debug(f"Delaying {ACCOUNT_DELAY_SECONDS} seconds before placing equity order")
                            time_module.sleep(ACCOUNT_DELAY_SECONDS)
                            current_market_order = self.market_order(self.current_ticker, int(send_qty))

                        # Mark trade state
                        self.number_of_trades += 1
                        trade_decider_function.set_open_trade(current_market_order)
                        trade_decider_function.open_trade_fill_price = candle.close
                        self.in_exit_mode = True
                        self.in_exit_mode_of_buy = trading_decision.decision_type == TradeDecisionType.BUY
                        self.traded_a_buy_today = self.in_exit_mode_of_buy
                        self.in_exit_mode_of_sell = trading_decision.decision_type == TradeDecisionType.SELL
                        self.traded_a_sell_today = self.in_exit_mode_of_sell
                        self.trade_entry_moment = self.time_reasonable
                        self.has_open_trade = True

                        # TakeProfitArea wiring
                        if algorithm_parameters.use_take_profit_area_logic:
                            current_op_range = self.get_opening_range_mainly_used(
                                algorithm_parameters.minutes_of_opening_range_used_for_exit_area_distance_after_trade
                            ).opening_ranges_by_date[self.time_reasonable.date()]

                            tp_initial = trade_decider_function.get_take_profit_initial_value(
                                candle,
                                current_op_range,
                                algorithm_parameters.coeficient_of_confirmation_cons_box_height_used_for_distance_to_take_profit_area,
                                algorithm_parameters.coeficient_of_opening_range_height_used_for_distance_to_take_profit_area,
                            )

                            if tp_initial:
                                take_profit_area_initial_value, take_profit_area_distance = tp_initial
                            else:
                                take_profit_area_initial_value = candle.close
                                take_profit_area_distance = candle.close * 0.0025

                            self.take_profit_area = TakeProfitArea(
                                start_time=self.time_reasonable,
                                end_time=self.time_reasonable + timedelta(minutes=300),
                                initial_value=take_profit_area_initial_value,
                                distance=take_profit_area_distance,
                                number_of_layers=algorithm_parameters.take_profit_area_number_of_layers,
                                min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer=algorithm_parameters.take_profit_area_min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer,
                                incremental_tunnel_closing_coeficient=algorithm_parameters.take_profit_area_incremental_tunnel_closing_coeficient,
                                traded_long=(trading_decision.decision_type == TradeDecisionType.BUY),
                                max_loss_threshold_percent=algorithm_parameters.max_loss_threshold_percent,
                                logger=self.logger if hasattr(self, "logger") else None,
                            )
                            self.take_profit_area.reset_state_for_new_trade()
                            self.take_profit_area.entry_price = candle.close

            else:
                # EXIT PATH
                # --- Safe current_candle for exit check ---
                current_candle_for_exit = self.all_daily_data.get(self.time_reasonable, None)

                if current_candle_for_exit is None:
                    self.logger.warning(f"[WARNING] No current_candle at {self.time_reasonable} — skipping should_exit_trade() for {trade_decider_function.name}")
                    return TradeDecision(
                        decision_type=TradeDecisionType.DONT_TRADE,
                        order_quantity=self.quantity_to_trade,
                        relevant_data={"note": "Skipped exit check due to missing candle"},
                        feature_data={},
                        is_opposite=False,
                    )

                # --- Should exit trade? ---
                should_exit_now = trade_decider_function.should_exit_trade(
                    is_market_open=self.is_current_market_open and not self.is_second_last_day_candle,
                    are_last_x_candles_closing_inside_the_exit_area_signal=self.are_last_x_candles_closing_inside_the_exit_area_signal,
                    a_15_min_box_just_closed_up_signal=self.a_15_min_box_just_closed_up_signal,
                    a_15_min_box_just_closed_down_signal=self.a_15_min_box_just_closed_down_signal,
                    current_candle=current_candle_for_exit,
                    is_current_candle_in_the_exit_area_signal=self.is_current_candle_in_the_exit_area_signal,
                    is_current_candle_in_the_take_profit_area=(
                        self.take_profit_area.is_candle_in_take_profit_area_long_enough(
                            candle=candle,
                            candle_moment=self.time_reasonable,
                            only_considering_closing_values=algorithm_parameters.only_considering_close_values_for_the_take_profit_area_logic,
                        ) if self.take_profit_area else False
                    ),
                )

                # --- Opposite box break check ---
                should_exit_due_to_opposite_box_break = False
                if self.trade_entry_moment:
                    minutes_since_entry = (self.time_reasonable - self.trade_entry_moment).total_seconds() / 60
                    if minutes_since_entry >= 5:
                        should_exit_due_to_opposite_box_break = self.should_exit_due_to_opposite_confirmation_box_break(
                            self.time_reasonable,
                            self.in_exit_mode_of_buy,
                            self.trade_entry_moment,
                        )

                # --- Perform exit ---
                if (should_exit_now and (not algorithm_parameters.must_be_worst_candle_in_exit_area_to_exit or self.is_last_candle_the_worst_price_in_exit_area())) or should_exit_due_to_opposite_box_break:
                    if not trade_decider_function.trade_decider_message_identifier:
                        trade_decider_function.generate_trade_decider_message_identifier()

                    if trade_decider_function.open_trade_fill_price is not None:
                        entry_price = trade_decider_function.open_trade_fill_price
                        exit_price = current_price
                        realized_return = (
                            (exit_price - entry_price) / entry_price
                            if self.in_exit_mode_of_buy
                            else (entry_price - exit_price) / entry_price
                        )
                        if not hasattr(self, "realized_returns_by_date"):
                            self.realized_returns_by_date = {}
                        if current_date not in self.realized_returns_by_date:
                            self.realized_returns_by_date[current_date] = []
                        self.realized_returns_by_date[current_date].append(realized_return)
                        if realized_return >= 0.001:
                            self.should_stop_trading_for_today = True
                            self.debug(f"Realized return of {realized_return:.2%} on {current_date} — halting further trades today.")

                    self.log_trade_exit(
                        trade_decider_function.name,
                        trade_decider_function.trade_decider_message_identifier,
                        current_price,
                    )
                    self.Debug(f"Delaying {ACCOUNT_DELAY_SECONDS} seconds before placing exit order")
                    time_module.sleep(ACCOUNT_DELAY_SECONDS)
                    self.liquidate()
                    self.has_open_trade = False
                    self.number_of_exits += 1
                    self.in_exit_mode = False
                    self.in_exit_mode_of_buy = False
                    self.in_exit_mode_of_sell = False
                    trade_decider_function.remove_open_trade()
                    self.take_profit_area = None

            # --- Return default ---
            return TradeDecision(
                decision_type=TradeDecisionType.DONT_TRADE,
                order_quantity=self.quantity_to_trade,
                relevant_data={},
                feature_data={},
                is_opposite=False,
            )

    def execute_trade_deciders(self, candle: MyCandle) -> None:

        intraday_trend_previous_day = self.intraday_trend_by_date[
            self.get_previous_open_market_day(how_many_before=1)
        ]
        intraday_trend_2_previous_day = self.intraday_trend_by_date[
            self.get_previous_open_market_day(how_many_before=2)
        ]

        opening_range_factories = [
            self.get_opening_range_mainly_used(x)
            for x in algorithm_parameters.minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal
        ]

        self.execute_specific_trade_decider(
            candle,
            self.high_volatility_box_breaker_trade_decider,
            self.is_1200_signal,
            self.is_d0_overnight_gap_indicating_volatility_signal,
            self.is_d1_overnight_gap_indicating_volatility_signal,
            self.is_d1_natr_indicating_volatility_signal,
            self.a_box_closed_higher_signal,
            self.a_box_closed_lower_signal,
            self.a_15_min_box_just_closed_down_signal,
            self.a_15_min_box_just_closed_up_signal,
            self.average_slope_2_days_percentile_bigger_than_90_threshold_signal,
            self.average_slope_2_days_percentile_bigger_than_75_threshold_signal,
            self.average_slope_1_day_percentile_bigger_than_95_threshold_signal,
            self.are_last_2_intraday_trend_directions_the_same_signal,
            self.is_last_intraday_trend_directions_upward_signal,
            self.is_last_intraday_trend_directions_downward_signal,
            self.has_x_closing_hlods_until_limit_signal,
            self.is_d1_trend_indicating_volatility_signal,
            self.is_d1_trend_lh_and_percentile_indicating_volatility_signal,
            intraday_trend_previous_day,
            intraday_trend_2_previous_day,
            self.d0_trend_lh_value,  # the d0 trendLH is actually the previous day, but it only gets the value updated at the end of the day
            self.d1_trend_lh_value,  # same for d2_trend_lh, TODO: fix this to avoid confusion later on
            self.d0_gap_magnitude_value,
            self.d1_gap_magnitude_value,
            self.two_day_average_gap_value,
            self.time_reasonable,
            current_candle=candle,
            first_candle_of_the_day=self.first_candle_of_the_day,
            moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far=self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
            moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far=self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
            opening_range_factories_to_consider=[
                self.get_opening_range_mainly_used(x)
                for x in algorithm_parameters.minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal
            ],
            
        )

    def update_opening_range_feature(self) -> None:

        current_date = self.time.date()

        for op_range, respective_cons_box_factory in [
            (
                self.opening_range_of_5_min,
                self.consolidation_box_factory_of_5_minutes_op,
            ),
            (
                self.opening_range_of_10_min,
                self.consolidation_box_factory_of_10_minutes_op,
            ),
            (
                self.opening_range_of_15_min,
                self.consolidation_box_factory_of_15_minutes_op,
            ),
        ]:

            opening_datetime = min(self.all_daily_data.keys())
            assert opening_datetime.day == self.time.day

            op_range.update_opening_range_feature(
                current_moment=self.time_reasonable,
                last_market_open_moment=self.last_market_open_moment,
                market_high_by_date=self.market_high_by_date,
                market_low_by_date=self.market_low_by_date,
                day_opening_price=self.all_daily_data[opening_datetime].open,
                use_midpoint_logic=algorithm_parameters.use_midpoint_logic_on_opening_range,
            )

            if (
                respective_cons_box_factory.consolidation_box_max_thickness_last_set_at
                is None
                or current_date
                > respective_cons_box_factory.consolidation_box_max_thickness_last_set_at
            ):

                if current_date in op_range.opening_ranges_by_date.keys():
                    respective_cons_box_factory.consolidation_box_max_thickness_last_set_at = (
                        current_date
                    )
                    respective_cons_box_factory.consolidation_box_max_thickness = (
                        get_average_opening_range_thickness(
                            op_range.opening_ranges_by_date, average_size=6
                        )
                    )

                    # UPDATE EXIT AREAS
                    if op_range == self.get_opening_range_mainly_used(
                        algorithm_parameters.minutes_of_opening_range_used_for_exit_area_distance_after_trade
                    ):

                        this_opening_range = op_range.opening_ranges_by_date[
                            current_date
                        ]

                        this_opening_range_thickness = abs(
                            this_opening_range.high - this_opening_range.low
                        ) * (
                            algorithm_parameters.percentage_of_opening_range_used_on_cross_op_range_risk_management
                            / 100
                        )

                        limit_moment = datetime(
                            year=current_date.year,
                            month=current_date.month,
                            day=current_date.day,
                            hour=algorithm_parameters.exit_area_beyond_opening_range_limit_hour,
                            minute=algorithm_parameters.exit_area_beyond_opening_range_limit_minute,
                        )
                        delta_until_limit = limit_moment - self.time_reasonable
                        minutes_until_limit = int(
                            delta_until_limit.total_seconds() / 60
                        )

                        end_of_day_limit_moment = datetime(
                            year=current_date.year,
                            month=current_date.month,
                            day=current_date.day,
                            hour=16,
                            minute=0,
                        )
                        delta_until_end_of_day_limit = (
                            end_of_day_limit_moment - self.time_reasonable
                        )
                        minutes_until_end_of_day_limit = int(
                            delta_until_end_of_day_limit.total_seconds() / 60
                        )

                        for i in range(minutes_until_end_of_day_limit + 1):
                            if False:
                                pass
                            else:
                                pass

    def update_atr_values(self) -> None:
        this_day_natr = self.my_natr.current.value
        self.natr_d0 = this_day_natr

        if this_day_natr is None:
            return

        this_day_regime_according_to_natr_classification = (
            classify_regime_according_to_natr(this_day_natr)
        )
        self.d0_regime_according_to_natr = (
            this_day_regime_according_to_natr_classification
        )

        this_day_regime_according_to_natr_probablistic = (
            calculate_probablistic_regime_classification_according_to_natr(
                this_day_natr
            )
        )
        self.d0_regime_according_to_natr_probablisticly = (
            this_day_regime_according_to_natr_probablistic
        )

        self.log_feature(
            f"# OF CLOSING HLODS FOR SIGNAL",
            algorithm_parameters.has_x_closing_hlods_until_limit_signal_enough_balance_for_a_signal,
        )
        self.log_feature(
            f"TIME LIMIT UNTIL SIGNAL OF CLOSING HLODS",
            f"{algorithm_parameters.has_x_closing_hlods_until_limit_signal_hour_of_limit}:{algorithm_parameters.has_x_closing_hlods_until_limit_signal_minute_of_limit}",
        )

        self.log_feature(
            f"D-0 NATR {algorithm_parameters.atr_days_period}", self.natr_d0
        )
        self.log_feature(
            f"REGIME CONCLUSION WRT D-0 NATR {algorithm_parameters.atr_days_period}",
            (
                self.d0_regime_according_to_natr.value
                if self.d0_regime_according_to_natr
                else self.d0_regime_according_to_natr
            ),
        )
        self.log_feature(
            f"PROB REGIME CONCLUSION WRT D-0 NATR {algorithm_parameters.atr_days_period}",
            self.d0_regime_according_to_natr_probablisticly,
        )
        self.log_feature(
            f"D-1 NATR {algorithm_parameters.atr_days_period}", self.natr_d1
        )
        self.log_feature(
            f"REGIME CONCLUSION WRT D-1 NATR {algorithm_parameters.atr_days_period}",
            (
                self.d1_regime_according_to_natr.value
                if self.d1_regime_according_to_natr
                else self.d1_regime_according_to_natr
            ),
        )
        self.log_feature(
            f"PROB REGIME CONCLUSION WRT D-1 NATR {algorithm_parameters.atr_days_period}",
            self.d1_regime_according_to_natr_probablisticly,
        )

    def get_opening_range_mainly_used(self, minutes: int) -> OpeningRangeFactory:
        if minutes == 5:
            return self.opening_range_of_5_min
        elif minutes == 10:
            return self.opening_range_of_10_min
        elif minutes == 15:
            return self.opening_range_of_15_min
        raise ValueError(f"Invalid minutes: {minutes}")

    def update_intraday_slope(self, candle: MyCandle) -> None:
        current_date = self.time.date()
        assert current_date in self.market_open_prices_by_date

        assert self.last_market_open_moment

        result: IntradayTrend = dx_intraday_trend_slope(
            open_value=self.market_open_prices_by_date[current_date],
            current_close_value=candle.close,
        )

        self.intraday_slopes_last_days.append(result.slope)
        if (
            len(self.intraday_slopes_last_days)
            > algorithm_parameters.intraday_slopes_last_days_max_amount
        ):
            overflow = (
                len(self.intraday_slopes_last_days)
                - algorithm_parameters.intraday_slopes_last_days_max_amount
            )
            self.intraday_slopes_last_days = self.intraday_slopes_last_days[overflow:]

        # get the classification
        percentile_rank = percentileofscore(
            [abs(x) for x in self.intraday_slopes_last_days],
            abs(result.slope),
        )

        if percentile_rank >= 100 - (
            algorithm_parameters.intraday_trend_classification_percentile * 2
        ):
            result.classification = (
                IntradayTrendClassification.UPWARD
                if result.slope > 0
                else IntradayTrendClassification.DOWNWARD
            )

        result.percentile = percentile_rank

        self.intraday_trend_by_date[current_date] = result

        self.intraday_slope_percentile_of_last_day = result.percentile

        self.intraday_trend_directions_of_last_2_days_are_the_same = (
            self.intraday_trend_direction_of_last_day == result.classification
        )

        self.intraday_trend_direction_of_last_day = result.classification

        previous_day_date = self.get_previous_open_market_day(how_many_before=1)
        previous_day_intraday_trend = (
            self.intraday_trend_by_date.get(previous_day_date)
            if previous_day_date
            else None
        )

        last_2_intraday_trends: list[IntradayTrend] = [
            x for x in [previous_day_intraday_trend, result] if x
        ]
        if last_2_intraday_trends:
            self.intraday_slope_percentile_average_of_last_2_days = sum(
                [x.percentile for x in last_2_intraday_trends]
            ) / len(last_2_intraday_trends)

    def on_end_of_algorithm(self) -> None:
        # log the number of trading days by accessing the number of market lows by date
        number_of_trading_days = len(self.market_low_by_date.keys())
        self.my_debug(f"TOTAL # TRADING DAYS: {number_of_trading_days}")
        self.my_debug(
            f"PROPORTION OF CASH TRADED: {algorithm_parameters.proportion_of_cash_to_trade}"
        )
        self.my_debug(f"TOTAL # TRADES: {self.number_of_trades}")
        self.my_debug(f"TOTAL # EXITS: {self.number_of_exits}")

    def erase_all_consolidation_boxes(self) -> None:
        for cons_box_factory in [
            self.consolidation_box_factory_of_5_minutes_op,
            self.consolidation_box_factory_of_10_minutes_op,
            self.consolidation_box_factory_of_15_minutes_op,
        ]:
            cons_box_factory.erase_all_consolidation_boxes()

    def close_all_open_consolidation_boxes(self, is_at_market_close: bool) -> None:

        meta_opening_range_factory = self.get_opening_range_mainly_used(
            algorithm_parameters.minutes_of_opening_range_used_for_classifying_consolidation_box_regarding_the_position_when_closing
        )

        if meta_opening_range_factory is None:
            self.logger.warning(f"[WARNING] No main opening range factory — skipping close_all_open_consolidation_boxes()")
            return

        meta_opening_range = meta_opening_range_factory.opening_ranges_by_date.get(self.time.date(), None)

        if meta_opening_range is None:
            self.logger.warning(f"[WARNING] No meta opening range for {self.time.date()} — skipping close_all_open_consolidation_boxes()")
            return

        for cons_box_factory in [
            self.consolidation_box_factory_of_5_minutes_op,
            self.consolidation_box_factory_of_10_minutes_op,
            self.consolidation_box_factory_of_15_minutes_op,
        ]:
            cons_box_factory.close_all_open_consolidation_boxes(
                all_daily_data=self.all_daily_data,
                closing_hod=self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
                closing_lod=self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
                current_moment=self.time_reasonable,
                meta_opening_range=meta_opening_range,
                considering_closing_candle_value_instead_of_highs_and_lows=algorithm_parameters.consolidation_box_using_closing_candle_values_instead_of_highs_and_lows,
                is_at_market_close=is_at_market_close,
            )

    def update_consolidation_boxes(self, is_at_market_close: bool) -> None:

        for cons_box_factory, respective_op_range in [
            (self.consolidation_box_factory_of_5_minutes_op, self.opening_range_of_5_min),
            (self.consolidation_box_factory_of_10_minutes_op, self.opening_range_of_10_min),
            (self.consolidation_box_factory_of_15_minutes_op, self.opening_range_of_15_min),
        ]:

            # --- Opening range for this box ---
            this_day_opening_range = respective_op_range.opening_ranges_by_date.get(self.time.date(), None)

            # --- Meta opening range for classification ---
            main_opening_range = self.get_opening_range_mainly_used(
                algorithm_parameters.minutes_of_opening_range_used_for_classifying_consolidation_box_regarding_the_position_when_closing
            )

            if main_opening_range is not None:
                meta_opening_range = main_opening_range.opening_ranges_by_date.get(self.time.date(), None)
                meta_opening_ranges_window_minutes = main_opening_range.opening_ranges_window_minutes
            else:
                meta_opening_range = None
                meta_opening_ranges_window_minutes = None

            # --- Current candle ---
            current_candle = self.all_daily_data.get(self.time_reasonable, None)

            if current_candle is None:
                self.logger.warning(f"[WARNING] No current_candle at {self.time_reasonable} — skipping update_consolidation_boxes() for {cons_box_factory}")
                continue

            # --- Update consolidation boxes ---
            cons_box_factory.update_consolidation_boxes(
                current_datetime=self.time_reasonable,
                current_candle=current_candle,
                closing_hod=self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
                closing_lod=self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
                this_day_opening_range=this_day_opening_range,
                meta_opening_range=meta_opening_range,
                all_daily_data=self.all_daily_data,
                last_market_open_moment=self.last_market_open_moment,
                meta_opening_ranges_window_minutes=meta_opening_ranges_window_minutes,
                considering_closing_candle_value_instead_of_highs_and_lows=algorithm_parameters.consolidation_box_using_closing_candle_values_instead_of_highs_and_lows,
                is_at_market_close=is_at_market_close,
            )

            # --- Update closing HOD/LOD logic ---
            cons_box_factory.update_consolidation_boxes_according_to_closing_hod_or_lod(
                current_datetime=self.time_reasonable,
                closing_hod=self.moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far,
                closing_lod=self.moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far,
            )

    def update_exit_area_when_buying_with_new_value(self, new_value: float) -> None:
        minutes_until_end_of_market_day = int(
            (self.last_candle_of_this_day - self.time_reasonable).total_seconds() / 60
        )

        for i in range(minutes_until_end_of_market_day + 1):
            target_moment = self.time_reasonable + timedelta(minutes=i)
            if target_moment in self.exit_area_when_buying.keys():
                self.exit_area_when_buying[target_moment] = max(
                    self.exit_area_when_buying[target_moment],
                    new_value,
                )
            else:
                self.exit_area_when_buying[target_moment] = new_value

    def update_exit_area_when_selling_with_new_value(self, new_value: float) -> None:
        minutes_until_end_of_market_day = int(
            (self.last_candle_of_this_day - self.time_reasonable).total_seconds() / 60
        )

        for i in range(minutes_until_end_of_market_day + 1):
            target_moment = self.time_reasonable + timedelta(minutes=i)
            if target_moment in self.exit_area_when_selling.keys():
                self.exit_area_when_selling[target_moment] = min(
                    self.exit_area_when_selling[target_moment],
                    new_value,
                )
            else:
                self.exit_area_when_selling[target_moment] = new_value

    def update_exit_area_with_new_candle(self, candle: MyCandle) -> None:
        if self.in_exit_mode_of_buy:
            if self.value_with_most_profit_when_buying is None:
                self.value_with_most_profit_when_buying = candle.close

            if self.current_difference_until_stop_loss is None:
                return  #  prevent crash

            if candle.close > self.value_with_most_profit_when_buying:
                self.value_with_most_profit_when_buying = candle.close

                new_value_of_exit_area = (
                    self.value_with_most_profit_when_buying
                    - self.current_difference_until_stop_loss
                )
                self.update_exit_area_when_buying_with_new_value(new_value_of_exit_area)

        elif self.in_exit_mode_of_sell:
            if self.value_with_most_profit_when_selling is None:
                self.value_with_most_profit_when_selling = candle.close

            if self.current_difference_until_stop_loss is None:
                return  #  prevent crash

            if candle.close < self.value_with_most_profit_when_selling:
                self.value_with_most_profit_when_selling = candle.close

                new_value_of_exit_area = (
                    self.value_with_most_profit_when_selling
                    + self.current_difference_until_stop_loss
                )
                self.update_exit_area_when_selling_with_new_value(new_value_of_exit_area)



    def is_last_candle_the_worst_price_in_exit_area(self) -> bool:
        """
        Determines if the current candle is the worst price seen inside the exit area.
        For buys, worst means lowest close. For sells, it means highest close.
        """
        lookback_minutes = algorithm_parameters.min_consecutive_closing_candles_on_exit_area_to_perform_exit
        lookback_moments = [
            self.time_reasonable - timedelta(minutes=i) for i in range(lookback_minutes)
        ]

        closes = []
        for moment in lookback_moments:
            candle = self.all_daily_data.get(moment)
            if candle is None:
                continue
            closes.append(candle.close)

        if len(closes) < lookback_minutes:
            return False

        current_close = self.all_daily_data[self.time_reasonable].close

        if self.in_exit_mode_of_buy:
            return current_close == min(closes)
        elif self.in_exit_mode_of_sell:
            return current_close == max(closes)

        return False

    def should_exit_due_to_opposite_confirmation_box_break(
        self,
        current_time: datetime,
        is_long: bool,
        trade_entry_moment: Optional[datetime],
    ) -> bool:
        for factory in self.consolidation_box_factories:
            if factory in [self.consolidation_box_factory_of_5_minutes_op,
                           self.consolidation_box_factory_of_10_minutes_op]:
                continue

            for box in factory.closed_consolidation_boxes:
                if box.box_break_inside is None:
                    continue

                if box.end_moment > current_time:
                    continue  # Box is from the future

                if trade_entry_moment and box.end_moment <= trade_entry_moment:
                    continue  # Skip boxes from before the trade

                # Check for opposite-direction break
                if is_long and box.box_break_inside != BoxBreakInsideClassification.DOWN:
                    continue
                if not is_long and box.box_break_inside != BoxBreakInsideClassification.UP:
                    continue

                # Box height and threshold calculation
                box_height = box.high - box.low
                threshold = (
                    box.low - 0.3 * box_height if is_long else box.high + 0.3 * box_height
                )

                # Scan post-break window
                post_break_window_start = box.end_moment + timedelta(minutes=1)
                post_break_window_end = box.end_moment + timedelta(minutes=100)

                t = post_break_window_start
                consecutive_bad_closes = 0

                while t <= current_time and t <= post_break_window_end:
                    if t in self.all_daily_data:
                        close = self.all_daily_data[t].close

                        if (is_long and close < threshold) or (not is_long and close > threshold):
                            consecutive_bad_closes += 1
                            if consecutive_bad_closes >= 2:
                                return True
                        else:
                            consecutive_bad_closes = 0  # reset if one good candle appears

                    t += timedelta(minutes=1)

        return False


    
    def on_data(self, slice: Slice) -> None:

        # the weirdest thing happens when backtesting:
        # if i backtest from before the 24th of December of 2018 onward,
        # there is a candle on Midnight of the 24th (right when the day starts)
        # that appears, OUT OF NOWHERE
        # i implemented a quick fix, but i cant really tell what is going on...
        if self.time.hour == 0 and self.time.minute == 0:
            return

        if self.time.date() in self.days_to_ignore:
            return

        if futures_trading_parameters.is_trading_futures:
            if self.selected_futures_contract is None:
                self.select_futures_contract(slice)

        if futures_trading_parameters.is_observing_futures:
            if (
                self.is_current_market_open is False
                and self.market_just_closed is False
            ):
                return

            assert self.future_contract is not None
            if self.future_contract not in slice.Bars:
                self.debug(f"No futures candle data on {self.time}")
                return
            bar: TradeBar = slice.bars[self.future_contract]

            new_candle = MyCandle(
                high=copy.deepcopy(bar.high),
                low=copy.deepcopy(bar.low),
                open=copy.deepcopy(bar.open),
                close=copy.deepcopy(bar.close),
                price=copy.deepcopy(bar.price),
                volume=copy.deepcopy(bar.volume),
                moment=bar.EndTime  # added moment
            )

        else:
            if not slice.bars.contains_key(self.current_ticker_object.symbol):
                return
            new_candle = MyCandle(
                high=copy.deepcopy(self.current_ticker_object.high),
                low=copy.deepcopy(self.current_ticker_object.low),
                open=copy.deepcopy(self.current_ticker_object.open),
                close=copy.deepcopy(self.current_ticker_object.close),
                price=copy.deepcopy(self.current_ticker_object.price),
                volume=copy.deepcopy(self.current_ticker_object.volume),
                moment=self.time  # added moment
            )

        self.time_reasonable = datetime(
            year=self.time.year,
            month=self.time.month,
            day=self.time.day,
            hour=self.time.hour,
            minute=self.time.minute,
            second=0,
        )
        self.last_candle_of_this_day = datetime(
            year=self.time.year,
            month=self.time.month,
            day=self.time.day,
            hour=16,
            minute=0,
            second=0,
        )
        is_last_day_candle = self.time_reasonable == self.last_candle_of_this_day
        is_second_last_day_candle = self.time_reasonable == (
            self.last_candle_of_this_day - timedelta(minutes=1)
        )
        self.is_second_last_day_candle = is_second_last_day_candle

        self.all_daily_data[self.time_reasonable] = new_candle

        self.all_1_minute_candles_by_time[self.time_reasonable] = new_candle

        if not self.first_candle_of_the_day:
            self.first_candle_of_the_day = new_candle

        self.update_moments_of_exceeding_closing_candles(new_candle)
        self.update_current_day_high_and_low(new_candle)
        self.update_consecutive_closing_candles_on_exit_area(new_candle)

        if self.in_exit_mode:
            self.update_exit_area_with_new_candle(new_candle)

        if self.is_current_market_open:
            if self.is_first_data_after_market_open:
                self.is_first_data_after_market_open = False
                self.deal_with_first_data_after_market_open(new_candle)

            self.update_opening_range_feature()

        self.update_consolidation_boxes(is_at_market_close=is_last_day_candle)
        if self.market_just_closed or is_last_day_candle:
            self.update_intraday_slope(new_candle)
            self.on_last_data_after_market_close(new_candle)
            self.close_all_open_consolidation_boxes(is_at_market_close=True)
            self.log_consolidation_boxes_feature()
            self.update_rolling_std()
            self.update_atr_values()

        self.execute_signals(new_candle)

        if self.is_warming_up is False:
            self.execute_trade_deciders(candle=new_candle)

            if self.has_open_trade and self.time.hour == 15 and self.time.minute == 58:
                self.Debug(f"Delaying {ACCOUNT_DELAY_SECONDS} seconds before forced exit at end of day")
                time_module.sleep(ACCOUNT_DELAY_SECONDS)
                self.Debug(f"FORCED EXIT at {self.time} due to end-of-day timing.")
                self.liquidate_position(reason="End of day forced exit at 15:58")

        self.log_current_candle(new_candle, self.current_symbol)

        if self.market_just_closed:
            self.memory_cleanup()
            self.market_just_closed = False

    def OnOrderEvent(self, order_event: OrderEvent):
        if order_event.Status == OrderStatus.Filled:
            if order_event.Direction == OrderDirection.Sell:
                self.last_exit_price = order_event.FillPrice
                self.Debug(f"[ORDER FILLED] Sell filled at {order_event.FillPrice}")
            elif order_event.Direction == OrderDirection.Buy:
                self.last_entry_price = order_event.FillPrice
                self.Debug(f"[ORDER FILLED] Buy filled at {order_event.FillPrice}")
