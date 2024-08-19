

import pandas as pd
import numpy as np

def detect_trend_periods(time_series):
    trend_periods = []
    current_trend = None
    trend_start_index = None

    # Iterate over the time series
    for i in range(1, len(time_series)):
        if current_trend is None:  # Start of a new trend period
            if time_series.iloc[i, 0] > time_series.iloc[i - 1, 0]:
                current_trend = 'Upward'
            elif time_series.iloc[i, 0] < time_series.iloc[i - 1, 0]:
                current_trend = 'Downward'
            else:
                continue  # No change, continue to the next iteration
            trend_start_index = time_series.index[i]

        # Check if the trend continues
        if current_trend == 'Upward' and time_series.iloc[i, 0] < time_series.iloc[i - 1, 0]:
            trend_periods.append((trend_start_index, time_series.index[i - 1], 'Upward'))
            current_trend = None
        elif current_trend == 'Downward' and time_series.iloc[i, 0] > time_series.iloc[i - 1, 0]:
            trend_periods.append((trend_start_index, time_series.index[i - 1], 'Downward'))
            current_trend = None

    # Check for an ongoing trend at the end of the time series
    if current_trend is not None:
        trend_periods.append((trend_start_index, time_series.index[-1], current_trend))

    return trend_periods