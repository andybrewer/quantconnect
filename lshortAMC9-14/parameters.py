# region imports
from AlgorithmImports import *
# endregion
from datetime import date, time
from pydantic import BaseModel


class AlgorithmParameters(BaseModel):
    # Given that we are performing an All-In on every trade (we buy all of the Equity we can),
    # this number below serves to set the proportion of the cash we want to use.
    # If proportion_of_cash_to_trade == 1 , then we are using all of our cash;
    # If proportion_of_cash_to_trade < 1 , then we are using part of our cash;
    # If proportion_of_cash_to_trade > 1 , then we are using all of our cash PLUS some borrowed cash (leverage);
    proportion_of_cash_to_trade: float

    coeficient_of_op_range_thickness_to_use_in_risk_management_distance_from_order_price: (
        float
    )
    percentile_of_slope_for_risk_management: int
    atr_days_period: int  # TODO: REMOVE THIS PARAM, ITS NOT BEING USED ANYWHERE
    windows_for_rolling_std: list[int]
    min_consecutive_closing_candles_on_exit_area_to_perform_exit: int
    intraday_trend_classification_percentile: float
    intraday_slopes_last_days_max_amount: int
    consolidation_box_min_minutes: int
    consolidation_boxes_max_overlap_threshold: float
    consolidation_box_post_minimum_minutes_tolerance_coeficient: float
    consolidation_box_using_closing_candle_values_instead_of_highs_and_lows: bool
    accumulated_candle_length_for_consolidation_box_exit_evaluation: int  # minutes
    has_x_closing_hlods_until_limit_signal_hour_of_limit: int
    has_x_closing_hlods_until_limit_signal_minute_of_limit: int
    has_x_closing_hlods_until_limit_signal_enough_balance_for_a_signal: int
    must_be_worst_candle_in_exit_area_to_exit: bool
    max_percent_distance_from_pre_signal_close_to_box: float
    max_percent_distance_from_last_exit_price_to_box: float  # or whatever value you want


    # this next feature is the moment where the exit area of crossing the
    # opening range ends (example: when the opening range is finished creating,
    # if there is a BUY Trade performed and the price goes below the Opening Range,
    # we liquidate to manage the risk. however, we only perform this until 10:30)
    exit_area_beyond_opening_range_limit_hour: int
    exit_area_beyond_opening_range_limit_minute: int
    incremental_tunnel_closing_exit_area_coeficient: float
    consolidation_box_thickness_percentage_for_exit_area_calc_after_break: float
    percentage_of_opening_range_used_on_cross_op_range_risk_management: (
        float  # CURRENTLY UNUSED
    )

    use_midpoint_logic_on_opening_range: bool
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_after_close: int
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_before_close: int
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_after_close: int
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_before_close: int
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_after_close: int
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_before_close: int
    how_many_days_to_keep_track_of_market_being_open: int

    # opening ranges selected
    minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal: list[int]
    minutes_of_opening_range_used_for_exit_area_distance_after_trade: int
    minutes_of_opening_range_used_for_classifying_consolidation_box_regarding_the_position_when_closing: (
        int
    )

    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_above: (
        float
    )
    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_below: (
        float
    )

    require_15_min_box_confirmation_on_the_trade_decider: bool
    ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation: (
        bool
    )
    allowed_minutes_overlap_of_start_of_confirmation_box_and_pre_signal_moment: (
        int  # number of minutes
    )

    start_date: date
    end_date: date
    cant_trade_beyond_this_time: time | None

    use_take_profit_area_logic: bool
    only_considering_close_values_for_the_take_profit_area_logic: bool

    coeficient_of_confirmation_cons_box_height_used_for_distance_to_take_profit_area: (
        float
    )
    coeficient_of_opening_range_height_used_for_distance_to_take_profit_area: float

    allow_consolidation_boxes_to_exist_on_opening_range_area: bool

    take_profit_area_min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer: (
        int
    )
    take_profit_area_number_of_layers: int
    take_profit_area_incremental_tunnel_closing_coeficient: float
    max_loss_threshold_percent: float  # e.g., 0.5 for 0.5% stop-loss
    percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal: float




#####
##### EDIT ONLY BELOW
#####

algorithm_parameters = AlgorithmParameters(
    start_date=date(year=2008, month=1, day=1),
    end_date=date(year=2008, month=1, day=31),
    proportion_of_cash_to_trade=1,
    coeficient_of_op_range_thickness_to_use_in_risk_management_distance_from_order_price=1,
    percentile_of_slope_for_risk_management=80,
    atr_days_period=14,
    windows_for_rolling_std=[15, 30, 60],
    min_consecutive_closing_candles_on_exit_area_to_perform_exit=2,
    intraday_trend_classification_percentile=16.5,
    intraday_slopes_last_days_max_amount=60,
    consolidation_box_min_minutes=45,
    consolidation_boxes_max_overlap_threshold=0.6,
    consolidation_box_post_minimum_minutes_tolerance_coeficient=1 / (60 * 4),
    consolidation_box_using_closing_candle_values_instead_of_highs_and_lows=False,
    accumulated_candle_length_for_consolidation_box_exit_evaluation=5,
    has_x_closing_hlods_until_limit_signal_hour_of_limit=10,
    has_x_closing_hlods_until_limit_signal_minute_of_limit=15,
    has_x_closing_hlods_until_limit_signal_enough_balance_for_a_signal=5,
    exit_area_beyond_opening_range_limit_hour=10,
    exit_area_beyond_opening_range_limit_minute=30,
    incremental_tunnel_closing_exit_area_coeficient=0.002,
    consolidation_box_thickness_percentage_for_exit_area_calc_after_break=0.65,
    percentage_of_opening_range_used_on_cross_op_range_risk_management=100,
    use_midpoint_logic_on_opening_range=False,
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    how_many_days_to_keep_track_of_market_being_open=10,
    minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal=[
        5
    ],  # [5 , 10 , 15] or [10, 15] or similar
    minutes_of_opening_range_used_for_exit_area_distance_after_trade=5,
    minutes_of_opening_range_used_for_classifying_consolidation_box_regarding_the_position_when_closing=5,
    require_15_min_box_confirmation_on_the_trade_decider=True,
    ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation=True,
    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_above=-5,
    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_below=-5,
    allowed_minutes_overlap_of_start_of_confirmation_box_and_pre_signal_moment=10,
    cant_trade_beyond_this_time=time(hour=15, minute=26),
    only_considering_close_values_for_the_take_profit_area_logic=False,
    use_take_profit_area_logic=False,
    coeficient_of_confirmation_cons_box_height_used_for_distance_to_take_profit_area=0.4,
    coeficient_of_opening_range_height_used_for_distance_to_take_profit_area=0.4,
    allow_consolidation_boxes_to_exist_on_opening_range_area=False,
    take_profit_area_min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer=1,
    take_profit_area_number_of_layers=3,
    take_profit_area_incremental_tunnel_closing_coeficient=0.002,
    max_loss_threshold_percent=0.5,  # or whatever % stop-loss you want to use
    percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal=0.25,
    must_be_worst_candle_in_exit_area_to_exit = False,
    max_percent_distance_from_pre_signal_close_to_box=0.4,
    max_percent_distance_from_last_exit_price_to_box=1.25,  # or whatever value you want




)

### BY ZACH

algorithm_parameters = AlgorithmParameters(
    start_date=date(year=2023, month=1, day=1),
    end_date=date(year=2024, month=7, day=31),
    proportion_of_cash_to_trade=1,
    coeficient_of_op_range_thickness_to_use_in_risk_management_distance_from_order_price=0.75,
    percentile_of_slope_for_risk_management=80,
    atr_days_period=14,
    windows_for_rolling_std=[15, 30, 60],
    min_consecutive_closing_candles_on_exit_area_to_perform_exit=9,
    intraday_trend_classification_percentile=16.5,
    intraday_slopes_last_days_max_amount=60,
    consolidation_box_min_minutes=28,
    consolidation_boxes_max_overlap_threshold=0.5,
    consolidation_box_post_minimum_minutes_tolerance_coeficient=0,
    consolidation_box_using_closing_candle_values_instead_of_highs_and_lows=True,
    accumulated_candle_length_for_consolidation_box_exit_evaluation=1,
    has_x_closing_hlods_until_limit_signal_hour_of_limit=12,
    has_x_closing_hlods_until_limit_signal_minute_of_limit=35,
    has_x_closing_hlods_until_limit_signal_enough_balance_for_a_signal=2,
    exit_area_beyond_opening_range_limit_hour=10,
    exit_area_beyond_opening_range_limit_minute=30,
    incremental_tunnel_closing_exit_area_coeficient=0,
    consolidation_box_thickness_percentage_for_exit_area_calc_after_break=0.65,
    percentage_of_opening_range_used_on_cross_op_range_risk_management=100,
    use_midpoint_logic_on_opening_range=True,
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_5_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_10_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_after_close=5,
    cons_box_15_min_minutes_to_consider_a_hod_or_lod_before_close=5,
    how_many_days_to_keep_track_of_market_being_open=10,
    minutes_of_opening_range_used_for_has_x_closing_hlods_until_limit_signal=[
        5
    ],  # [5 , 10 , 15] or [10, 15] or similar
    minutes_of_opening_range_used_for_exit_area_distance_after_trade=15,
    minutes_of_opening_range_used_for_classifying_consolidation_box_regarding_the_position_when_closing=10,
    require_15_min_box_confirmation_on_the_trade_decider=True,
    ignore_pre_signal_avoiding_conditions_in_case_of_requiring_15_min_box_confirmation=True,
    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_above=-1,
    coeficient_of_distance_between_pre_signal_candle_and_confirmation_candle_regarding_height_of_cons_box_below=0.001,
    allowed_minutes_overlap_of_start_of_confirmation_box_and_pre_signal_moment=60,
    cant_trade_beyond_this_time=time(hour=15, minute=34),
    use_take_profit_area_logic=True,
    only_considering_close_values_for_the_take_profit_area_logic=True,
    coeficient_of_confirmation_cons_box_height_used_for_distance_to_take_profit_area=0.025,
    coeficient_of_opening_range_height_used_for_distance_to_take_profit_area=20,
    allow_consolidation_boxes_to_exist_on_opening_range_area=False,
    take_profit_area_min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer=20,
    take_profit_area_number_of_layers=1,
    take_profit_area_incremental_tunnel_closing_coeficient=0,
    max_loss_threshold_percent=0.35,  # or whatever % stop-loss you want to use
    percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal=0.25,
    must_be_worst_candle_in_exit_area_to_exit = False,
    max_percent_distance_from_pre_signal_close_to_box=1.25,
    max_percent_distance_from_last_exit_price_to_box=1.25,  # or whatever value you want



)
