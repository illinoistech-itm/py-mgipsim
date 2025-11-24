"""
Utilities and question functions for anomaly-detection QA over CGM data.

Assumptions:
- Sampling: 5-minute cadence; dataframe index = point index (0-based).
- Time units in intervals: minutes. We export intervals as {"start": int, "end": int} in MINUTES.
- Columns required: "IG (mmol/L)", "faults_label".
- Derived columns: BG (mg/dL), day (0-based), minute_of_day (0..1435), hour (0..23).
"""

import pandas as pd
import numpy as np

def preprocess_df(df):
    """
    Add derived columns for downstream rules.

    - BG: mg/dL = IG (mmol/L) * 18
    - day: 0-based day index assuming 288 points/day (5-min sampling)
    - minute_of_day: minute within day (0..1435)
    - hour: hour within day (0..23)

    Args:
        df (pd.DataFrame): Must contain column "IG (mmol/L)".

    Returns:
        pd.DataFrame: Copy/view with added columns.
    """
    df["BG"] = df["IG (mmol/L)"] * 18
    df['day'] = df.index // 288  
    df['minute_of_day'] = (df.index % 288) * 5  
    df['hour'] = df['minute_of_day'] // 60
    return df

def get_intervals_by_label(df, labels):
    """
    Build contiguous intervals for rows whose faults_label ∈ labels.

    Interval semantics:
      - Input contiguity is at the POINT level (adjacent indices).
      - Output {"start","end"} are in MINUTES (index * 5).
      - Both start/end are inclusive.

    Args:
        df (pd.DataFrame)
        labels (List[str] or Set[str])

    Returns:
        List[Dict[str,int]]: [{"start": int, "end": int}, ...] in minutes.
    """
    mask = df["faults_label"].isin(labels)
    indices = df.index[mask].tolist()
    if not indices:
        return []
    intervals = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            intervals.append({"start": start*5, "end": prev*5}) 
            start = prev = idx
    intervals.append({"start": start*5, "end": prev*5})  
    return intervals

def get_points_by_label(df, labels):
    """
    Return point indices (0-based) where faults_label ∈ labels.

    Args:
        df (pd.DataFrame)
        labels (str or Iterable[str])

    Returns:
        List[int]: point indices (NOT minutes).
    """
    if isinstance(labels, str):
        labels = [labels]
    return df[df["faults_label"].isin(labels)].index.tolist()

def get_intervals_from_bool_mask(mask):
    """
    Build contiguous intervals from a boolean mask over the dataframe index.

    - Contiguity at the POINT level (adjacent indices).
    - Output in MINUTES (index * 5), inclusive ends.

    Args:
        mask (pd.Series[bool]): indexed like df.index.

    Returns:
        List[Dict[str,int]]: [{"start": int, "end": int}, ...] in minutes.
    """
    indices = mask[mask].index.tolist()
    if not indices:
        return []
    intervals = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            intervals.append({"start": start*5, "end": prev*5})
            start = prev = idx
    intervals.append({"start": start*5, "end": prev*5})
    return intervals


def extract_intervals_in_range(df, start_idx, end_idx, labels):
    """
    Slice by POINT indices [start_idx, end_idx) then aggregate by faults_label.

    Args:
        df (pd.DataFrame)
        start_idx (int): point-index (NOT minutes), inclusive
        end_idx (int): point-index (NOT minutes), exclusive
        labels (Iterable[str])

    Returns:
        List[Dict[str,int]]: intervals in minutes.
    """
    sub_df = df.iloc[start_idx:end_idx].copy()
    return get_intervals_by_label(sub_df, labels)


def extract_intervals_in_day(intervals, periods):
    """
    Select or truncate intervals that overlap periods that given in a day.

    Args:
        intervals: list of dicts with {'start': int, 'end': int} in minutes.
        periods: a list of tuple (start_min, end_min) within a day, e.g. (1320, 360) for 22:00–06:00.

    Returns:
        List of dicts with {'start', 'end'} covering full night periods that overlap.
    """

    DAY = 1440

    # --- Step 1: Normalize periods into non-wrapping absolute ranges ---
    # For each daily period (ns, ne):
    #   - If ns < ne → simple interval [ns, ne]
    #   - If ns > ne → split into [ns, DAY] and [0, ne]
    normalized_periods = []
    for ns, ne in periods:
        if ns < ne:
            normalized_periods.append((ns, ne))
        else:  # wraps midnight
            normalized_periods.append((ns, DAY))
            normalized_periods.append((0, ne))

    selected = []

    # --- Step 2: For each interval, map each normalized period to the interval's day(s) ---
    for iv in intervals:
        abs_start = iv["start"]
        abs_end = iv["end"]

        # It may span multiple days, so handle day offsets explicitly
        # Compute the day index of the interval start
        base_day = abs_start // DAY

        # We will check period windows for both the start day and possibly the next day
        # because intervals may span midnight.
        candidate_days = {base_day, base_day - 1, base_day + 1}

        for day in candidate_days:
            day_offset = day * DAY

            for ps, pe in normalized_periods:
                # Map the period into absolute time on this day
                p_start_abs = day_offset + ps
                p_end_abs = day_offset + pe

                # --- Step 3: Compute intersection ---
                inter_start = max(abs_start, p_start_abs)
                inter_end = min(abs_end, p_end_abs)

                if inter_start < inter_end:
                    selected.append({
                        "start": inter_start,
                        "end": inter_end
                    })

    return selected


def extract_hypoglycemia_intervals(df, start_idx=None, end_idx=None, nocturnal=False, min_duration=None):
    """
    Build BG<70 intervals, with optional filters.

    Args:
        df (pd.DataFrame)
        start_idx (int|None): slice start in POINT indices, inclusive.
        end_idx (int|None): slice end in POINT indices, exclusive.
        nocturnal (bool): if True, keep only 00:00–06:00 intervals.
        min_duration (int|None): minimum DURATION in POINTS (e.g., 12 points = 60 min).

    Returns:
        List[Dict[str,int]]: intervals in minutes.
    """
    sub_df = df if start_idx is None else df.iloc[start_idx:end_idx].copy()
    mask = sub_df["BG"] < 70
    intervals = get_intervals_from_bool_mask(mask)

    if nocturnal:
        night_period = [(0, 360)]  # 00:00–06:00 22:00–06:00: (1320, 1440), (0, 360)
        # include any interval that overlaps with a night period
        # intervals = [i for i in intervals if all((x % 1440) < 360 for x in range(i["start"], i["end"] + 1))]
        intervals = extract_intervals_in_day(intervals, night_period)
    if min_duration:
        intervals = [i for i in intervals if (i["end"] - i["start"]) >= min_duration]

    return intervals


def compute_missing_signal_percentage_ad1(df):
    """
    ad_1: What percentage of my CGM data is missing?
    answer_generation_rule: Calculate the percentage of missing CGM points (NaNs in IG).
    answer_instruction: Return the percentage of missing CGM values (as float, not string). Use (missing_points / total_points) * 100.
    answer_type: float (percentage)
    metric: sMAPE
    """
    total_points = len(df)
    missing_points = df['IG (mmol/L)'].isna().sum()
    if total_points == 0:
        return np.nan
    percentage = (missing_points / total_points) * 100
    return round(percentage, 2)


def compute_missing_signal_days_ad2(df, threshold=0.3):
    """
    ad_2: Were there any days when data quality was too poor to interpret trends?
    answer_generation_rule: Return all days where more than 30% of data is missing (IG is NaN).
    answer_instruction: Return a list of day indices (1-based) where missingness exceeds 30% of the day‘s readings.
    answer_type: list of int (e.g., days)
    metric: Accuracy
    """
    result = []
    for day_id, group in df.groupby('day'):
        total = len(group)
        missing = group['IG (mmol/L)'].isna().sum()
        if total and (missing / total) > threshold:
            result.append(int(day_id) + 1)
    return result

def extract_missing_signal_intervals_ad3(df):
    """
    ad_3: Is there any missingness in the data?
    answer_generation_rule: Find all continuous time intervals of missing CGM data (IG is NaN).
    answer_instruction: Return a list of time intervals where IG is NaN, each with a 'start' and 'end' index.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_from_bool_mask(df['IG (mmol/L)'].isna())


def extract_negative_spike_points_ad4(df):
    """
    ad_4: Did the CGM show an implausible drop in glucose?
    answer_generation_rule: Intervals with faults_label == "negative_spike".
    answer_instruction:  Return a list of time intervals where the blood glucose readings exhibit abrupt drops of approximately 60 mg/dL or more compared to the previous value.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["negative_spike"])

def extract_abnormal_fault_intervals_ad5(df):
    """
    ad_5: Are there any artifacts in the CGM data?
    answer_generation_rule: Intervals where faults_label is one of the injected CGM-related artifact types.
    answer_instruction: Return a list of time intervals where the blood glucose data exhibits abnormal patterns such as sudden spikes or drops, missing values, repeated readings, or unexpected insulin behaviors.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    valid_labels = [
        "negative_spike", "missing_signal", "repeated_reading",  "max_reading", "min_reading",
        "negative_bias", "positive_spike", "positive_bias", "repeated_episode", "zero_reading"
    ]
    # Not CGM-related faults: "positive_basal", "false_bolus", "negative_basal", "unknown_under", "min_basal", "unknown_stop", "false_meal", "max_basal"

    return get_intervals_by_label(df, valid_labels)

def extract_repeated_reading_intervals_ad6(df):
    """
    ad_6: At what time intervals do prolonged repeated CGM readings occur that may indicate a data logging error?
    answer_generation_rule: Time intervals with faults_label == "repeated_reading"
    answer_instruction: Scan the dataset for consecutive timestamps with identical glucose values. Return a list of time intervals where these repeated values persist over multiple points and prolonged duration, as they may indicate potential data logging errors.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["repeated_reading"])

def extract_prolonged_repeated_intervals_ad7(df, min_duration=36):
    """
    ad_7: Did the sensor give flat-line readings for a prolonged time?
    answer_generation_rule: Time intervals with faults_label == "repeated_reading" and duration ≥ 180
    answer_instruction: Return a list of time intervals where the blood glucose readings remain flat (unchanged) for at least 36 consecutive time points(5-minutes sampled).
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    intervals = get_intervals_by_label(df, ["repeated_reading"])
    return [interval for interval in intervals if (interval["end"] - interval["start"]) // 5 + 1 >= min_duration]

def extract_sensor_dislodged_intervals_ad8(df):
    """
    ad_8: Was the CGM sensor dislodged during this period?
    answer_generation_rule: Time intervals with faults_label == "repeated_reading" or "zero_reading" or "positive_spike" or "negative_spike"
    answer_instruction: Return a list of time intervals where the blood glucose readings either remain flat for an extended period or are recorded as exactly zero, which may indicate a dislodged or malfunctioning sensor.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["repeated_reading", "zero_reading", "positive_spike", "negative_spike"])

def extract_spike_intervals_ad9(df):
    """
    ad_9: Is there an abrupt spike that could be a calibration error?
    answer_generation_rule: Intervals with faults_label == "positive_bias" or "negative_bias"
    answer_instruction: Return a list of time intervals where the blood glucose readings exhibit abrupt increases or decreases compared to the previous reading, sustained across consecutive points.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["positive_bias", "negative_bias"])

def extract_high_readings_intervals_ad10(df):
    """
    ad_10: Did my CGM report sudden high readings that quickly normalized without insulin?
    answer_generation_rule: Intervals with faults_label == "positive_spike" or "negative_spike"
    answer_instruction: Return a list of time intervals where the blood glucose readings exhibit sudden and sharp increases or decreases of approximately 60 mg/dL or more compared to the previous value.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    # df = df.iloc[:-1]
    return get_intervals_by_label(df, ["positive_spike", "negative_spike"])

def extract_last_week_drift_intervals_ad11(df):
    """
    ad_11: Does the last week's data show signs of sensor drift?
    answer_generation_rule: Extract data from 24th to 30th day; Intervals with faults_label in: positive_spike or negative_spike or repeated_reading or zero_reading
    answer_instruction: From day 24 to day 30, return a list of time intervals where the blood glucose readings exhibit abnormalities, including sudden large changes, prolonged flat-line values, or consecutive zero readings, which may indicate sensor instability or drift.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    # df = df.iloc[:-1]
    return extract_intervals_in_range(df, 23 * 288, 30 * 288,
        ["positive_spike", "negative_spike", "repeated_reading", "zero_reading"])

def extract_weekend_missing_intervals_ad12(df):
    """
    ad_12: Was there a data dropout in the last weekend (28th and 29th day)?
    answer_generation_rule: Extract 28th and 29th day; Periods between data points with label: [missing_start, missing_end]
    answer_instruction: From day 28 to 29, return a list of time intervals where blood glucose readings are missing (e.g., no value recorded), indicating possible data dropout.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_intervals_in_range(df, 27 * 288, 29 * 288, ["missing_signal"])

def extract_glucose_spikes_last_week_ad13(df):
    """
    ad_13: Were there any unexpected glucose spikes in the last week?
    answer_generation_rule: Extract data from 24th to 30th day; Timestamps of data points with label: positive_spike or negative_spike
    answer_instruction: From day 24 to 30, return a list of time intervals where the blood glucose readings show abrupt increases or decreases of approximately 60 mg/dL or more compared to the previous reading.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    # df = df.iloc[:-1]
    return extract_intervals_in_range(df, 23 * 288, 30 * 288, ["positive_spike", "negative_spike"])

def extract_rapid_drop_points_ad14(df):
    """
    ad_14: Was there a rapid drop in glucose in the 14th day?
    answer_generation_rule: Extract data in day 14; Time intervals with label: negative_spike
    answer_instruction: In day 14, return a list of time intervals where the blood glucose readings suddenly large drop compared to the previous value.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_intervals_in_range(df, 13 * 288, 14 * 288, ["negative_spike"])

def extract_glucose_drop_points_ad15(df):
    """
    ad_15: Was there a glucose drop of more than 50 mg/dL at any time in the first week?
    answer_generation_rule: Extract data from 1st to 7th day; Time intervals with label: negative_spike
    answer_instruction: From day 1 to 7, return a list of time intervals where the blood glucose readings show a sudden drop of approximately 50 mg/dL or more compared to the previous reading.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_intervals_in_range(df, 0, 7 * 288, ["negative_spike"])

def extract_morning_rise_points_ad16(df):
    """
    ad_16: Did my glucose rise quickly this morning?
    answer_generation_rule: Extract 30th day's data in the morning (6 am - 12 pm); time intervals with label: positive_spike
    answer_instruction: From 6:00 to 12:00 on day 30, return the time intervals where the blood glucose readings show a sudden increase of approximately 60 mg/dL or more compared to the previous value.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_intervals_in_range(df, 29 * 288 + 72, 29 * 288 + 144, ["positive_spike"])

def extract_most_recent_hypoglycemia_ad17(df):
    """
    ad_17: When did my most recent episode of hypoglycemia occur?
    answer_generation_rule: Return the last interval where BG < 70.
    answer_instruction: Return one time interval representing the most recent episode where blood glucose levels dropped below 70 mg/dL.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    intervals = extract_hypoglycemia_intervals(df)
    return [intervals[-1]] if intervals else []

def extract_nocturnal_hypoglycemia_ad18(df):
    """
    ad_18: Did I have nocturnal hypoglycemia?
    answer_generation_rule: Return intervals from 00:00 to 06:00 where BG < 70.
    answer_instruction: Return all time intervals between 00:00 and 06:00 where blood glucose was continuously below 70 mg/dL.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_hypoglycemia_intervals(df, nocturnal=True)

def extract_prolonged_nocturnal_hypoglycemia_ad19(df):
    """
    ad_19: Did I have a prolonged hypoglycemia episode overnight?
    answer_generation_rule: Return intervals from 00:00 to 06:00 where BG < 70 and duration ≥ 12 time points (each point = 5 minutes).
    answer_instruction: Return all time intervals between 00:00 and 06:00 where blood glucose was below 70 mg/dL for at least 60 minutes (i.e., 12 consecutive 5-minute points).
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return extract_hypoglycemia_intervals(df, nocturnal=True, min_duration=12)

def extract_last_week_hypoglycemia_intervals_ad20(df):
    """
    ad_20: Did I have a severe hypo event in the last week?
    answer_generation_rule: Return intervals with BG < 70 from day 24–30.
    answer_instruction: From day 24 to 30, return all time intervals where blood glucose was continuously below 70 mg/dL.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    # df = df.iloc[:-1]
    return extract_hypoglycemia_intervals(df, 23 * 288, 30 * 288)

def extract_back_to_back_hypo_hyper_ad21(df):
    """
    ad_21: Did I experience back-to-back hypo and hyper events?
    answer_generation_rule: If a hyperglycemia interval begins within 2 hours after a hypoglycemia interval, merge them.
    answer_instruction: Return time intervals that combine a hypoglycemia episode (BG < 70) immediately followed by a hyperglycemia episode (BG > 180) within 2 hours.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    hypo = get_intervals_from_bool_mask(df["BG"] < 70)
    hyper = get_intervals_from_bool_mask(df["BG"] > 180)
    result = []
    for h in hypo:
        for g in hyper:
            if 0 < g["start"] - h["start"] <= 120:
                result.append({"start": h["start"], "end": g["end"]})
                break
    return result

def extract_last_hypoglycemia_duration_ad22(df):
    """
    ad_22: How long was my last episode of hypoglycemia?
    answer_generation_rule: Length of the most recent period of data points with a hypoglycemia label: hypoglycemia_end - hypoglycemia_start
    answer_instruction: Return the duration, in number of time points, of the most recent episode where blood glucose was continuously below 70 mg/dL. Duration = end - start + 1.
    answer_type: int
    metric: sMAPE
    """
    intervals = get_intervals_from_bool_mask(df["BG"] < 70)
    return np.nan if not intervals else intervals[-1]["end"] - intervals[-1]["start"] + 1

def extract_hyperglycemia_starts_day25_ad23(df):
    """
    ad_23: At what time did I enter hyperglycemia on the 25th day?
    answer_generation_rule: Extract all intervals on the 25th day where BG > 180.
    answer_instruction: Return all intervals [start, end] on day 25 where blood glucose readings exceeded 180 mg/dL.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    window_df = df.iloc[24 * 288 : 25 * 288]
    intervals = get_intervals_from_bool_mask(window_df["BG"] > 180)
    return intervals

def extract_spikes_day27_lunch_ad24(df):
    """
    ad_24: Was there a significant glucose spike after lunch last Friday (the 27th day)?
    answer_generation_rule: Extract the data from 12 am to 6 pm on the 27th day. Intervals with faults_label == positive_spike
    answer_instruction: From 00:00 to 18:00 on day 27, return all time intervals where the blood glucose readings show a sudden increase of approximately 60 mg/dL or more compared to the previous value, sustained across consecutive points.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    window_df = df.iloc[26 * 288:26 * 288 + 216]
    return get_intervals_by_label(window_df, ["positive_spike"])

def count_total_hypoglycemia_events_ad25(df):
    """
    ad_25: How many times did I experience hypoglycemia?
    answer_generation_rule: Count the number of distinct hypoglycemia episodes where BG < 70.
    answer_instruction: Return the total number of distinct episodes where blood glucose was continuously below 70 mg/dL.
    answer_type: int
    metric: sMAPE
    """
    return len(get_intervals_from_bool_mask(df["BG"] < 70))

def count_last_week_hypoglycemia_events_ad26(df):
    """
    ad_26: How many times did I have hypo events in the last week?
    answer_generation_rule: Extract data from the 24th to the 30th day. Count the number of hypoglycemia events (continuous period counts as 1 event)
    answer_instruction: Extract data from the last 7 days, return the number of distinct episodes where blood glucose readings were continuously below 70 mg/dL. Each continuous stretch counts as one event.
    answer_type: int
    metric: sMAPE
    """
    if len(df) <= 7 * 288:
        window_df = df.iloc[:]
    else:
        window_df = df.iloc[-7 * 288:]
    return len(get_intervals_from_bool_mask(window_df["BG"] < 70))

def count_last_week_hyperglycemia_events_ad27(df):
    """
    ad_27: How many times did I have hyperglycemia events in the last week?
    answer_generation_rule: Extract data from the last 7 days. Count the number of distinct hyperglycemia events (BG > 180).
    answer_instruction: Extract data from the last 7 days, return the number of distinct episodes where blood glucose readings were continuously above 180 mg/dL.
    answer_type: int
    metric: sMAPE
    """
    if len(df) <= 7 * 288:
        window_df = df.iloc[:]
    else:
        window_df = df.iloc[-7 * 288:]
    return len(get_intervals_from_bool_mask(window_df["BG"] > 180))

def extract_prolonged_hyperglycemia_intervals_ad28(df):
    """
    ad_28: Was I in hyperglycemia for more than 4 hours last week?
    answer_generation_rule: Extract data from the last 7 days (day 24 to day 30). Identify intervals where BG > 180 persists for at least 48 consecutive data points (each representing a 5-minute interval, i.e., 4 hours).
    answer_instruction: Return all time intervals from the last 7 days, where blood glucose remained above 180 mg/dL for at least 48 consecutive data points (5-minute intervals).
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    if len(df) <= 7 * 288:
        window_df = df.iloc[:]
    else:
        window_df = df.iloc[-7 * 288:]
    intervals = get_intervals_from_bool_mask(window_df["BG"] > 180)
    return [i for i in intervals if i["end"] - i["start"] + 1 >= 48*5]

def compute_avg_recovery_time_from_hyper_ad29(df):
    """
    ad_29: How long did it generally take to reach target range after a prolonged hyperglycemia episode? 
    answer_generation_rule: Average value of the time BG returns to TIR minus the time the hyperglycemia event start
    answer_instruction: Examine each prolonged hyperglycemia episode (lasting at least 12 consecutive data points above 180 mg/dL). For each episode, calculate the duration from the start of hyperglycemia until glucose first returns to the target range (≤180 mg/dL). Return the average of these durations, reported in minutes, as the typical recovery time.
    answer_type: int
    metric: sMAPE
    """
    intervals = [i for i in get_intervals_from_bool_mask(df["BG"] > 180) if i["end"] - i["start"] >= 12*5]
    recovery_times = []
    for interval in intervals:
        # recovery_times.append(interval["end"] - interval["start"] + 5)
        start_idx = interval["start"] // 5
        recovery = df.iloc[start_idx:].index[df["BG"].iloc[start_idx:] <= 180].tolist()
        if recovery:
            duration_points = recovery[0] - start_idx
            recovery_times.append(duration_points * 5)

    return round(sum(recovery_times) / len(recovery_times), 2) if recovery_times else np.nan


def compute_last_hypo_recovery_time_ad30(df):
    """
    ad_30: When did I recover from the last hypoglycemic event?
    answer_generation_rule: Find the last hypoglycemia interval (BG < 70) and compute how long it took to recover.
    answer_instruction: Return the number of time points between the start of the most recent hypoglycemia episode (BG < 70)
                        and the first reading ≥ 70 mg/dL afterward.
    answer_type: int
    metric: sMAPE
    """
    intervals = get_intervals_from_bool_mask(df["BG"] < 70)
    if not intervals:
        return np.nan   
    start = intervals[-1]["start"]
    end = intervals[-1]["end"]
    recovery = df.iloc[end + 1:].index[df["BG"].iloc[end + 1:] >= 70].tolist()
    return recovery[0] - start if recovery else np.nan

def find_day_with_most_out_of_range_ad31(df):
    """
    ad_31: What day was my glucose control the most out of range?
    answer_generation_rule: Count the number of hypo and hyper labels each day. The day has the max value
    answer_instruction: Return the 1-indexed day number that had the highest total number of blood glucose readings outside the 70–180 mg/dL range.
    answer_type: int
    metric: Accuracy
    """
    counts = df[(df["BG"] < 70) | (df["BG"] > 180)]['day'].value_counts()
    return np.nan if counts.empty else int(counts.idxmax()) + 1

def compute_longest_hyperglycemia_duration_ad32(df):
    """
    ad_32: What was my longest time spent (in minutes) in hyperglycemia?
    answer_generation_rule: Compute the duration of all hyperglycemia episodes and return the maximum.
    answer_instruction: Examine all periods where glucose readings remain continuously above 180 mg/dL. For each period, calculate its duration in minutes from the first to the last timestamp. Return the longest of these durations as the maximum time spent in a single hyperglycemia episode.
    answer_type: int
    metric: sMAPE
    """
    intervals = get_intervals_from_bool_mask(df["BG"] > 180)
    return max([(i["end"] - i["start"]) + 1 for i in intervals], default=0)

def extract_negative_spikes_after_lunch_before_day20_ad33(df):
    """
    ad_33: My glucose dropped unusually fast after lunch on the 20th day. Has this happened before, and if so, when?
    answer_generation_rule: Extract negative_spike label within 12:00–14:00.
    answer_instruction: Return all time intervals between 12:00 and 14:00 on where blood glucose readings dropped abruptly by approximately 60 mg/dL or more compared to the previous reading.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    mask = df["minute_of_day"].between(720, 840) & (df["faults_label"] == "negative_spike")
    return get_intervals_by_label(df[mask], ["negative_spike"])

def count_alert_events(df):
    """
    Count distinct out-of-range episodes (BG<70 or BG>180).

    - Two consecutive points belong to the same episode.
    - Returns number of episodes over the given slice of df.

    Args:
        df (pd.DataFrame)

    Returns:
        int
    """
    indices = df.index[(df["BG"] < 70) | (df["BG"] > 180)]
    if indices.empty: return 0
    count = 1
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            count += 1
    return count

def compare_alerts_first_last_week_ad34(df):
    """
    ad_34: Has the number of alerts increased in the last week compared to the first week?
    answer_generation_rule: Count the number of BG out-of-range events (hypo/hyper) in first and last week. If more in the last week, return "yes", else "no".
    answer_instruction: Compare the total number of distinct alert episodes (continuous readings outside 70–180 mg/dL) during day 1–7 and day 24–30. Return "yes" if the number increased in the last week, otherwise return "no".
    answer_type: "yes" or "no"
    metric: Accuracy
    """
    return "yes" if count_alert_events(df.iloc[-7 * 288:]) > count_alert_events(df.iloc[:7 * 288]) else "no"


def extract_new_hypo_hours_last_week_ad35(df):
    """
    ad_35: Was there a new time of day when I experienced hypoglycemia in the last week?
    answer_generation_rule: Compare hours of hypoglycemia (BG < 70) between the first 3 weeks and the last week. Return hours that only occurred in the last week.
    answer_instruction: Return a sorted list of hour values (0–23) during which hypoglycemia (BG < 70 mg/dL) occurred in day 24–30 but did not occur during day 1–21.
    answer_type: list of int
    metric: Accuracy
    """
    early_subset = df.iloc[:-7*288]
    last_week_subset = df.iloc[-7*288:]
    early = early_subset[early_subset["BG"] < 70]["hour"]
    late = last_week_subset[last_week_subset["BG"] < 70]["hour"]
    return sorted(set(late) - set(early))

def extract_post_lunch_glucose_exceeds_220_ad36(df):
    """
    ad_36: My post-lunch glucose peaked at 220 mg/dL in the 24th day. What other days this month had similar unusual responses?
    answer_generation_rule: Extract intervals where BG > 220 between 12:00–14:00 across all days.
    answer_instruction: Return all time intervals where blood glucose readings exceeded 220 mg/dL between 12:00 and 14:00 on any day.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    mask = (df["minute_of_day"].between(720, 840)) & (df["BG"] > 220)
    return get_intervals_from_bool_mask(mask)

def detect_rollercoaster_pattern_ad37(df):
    """
    ad_37: Did I experience a rollercoaster pattern (rapid high-low cycles)?
    answer_generation_rule: Identify sequences of [hyper, hypo, hyper] within a 2-hour span.
    answer_instruction: Return all time intervals where blood glucose transitioned from hyperglycemia (BG > 180) to hypoglycemia (BG < 70), then back to hyperglycemia within a 2-hour span between each transition.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    events = [(i, "hyper") if bg > 180 else (i, "hypo") for i, bg in df["BG"].items() if bg < 70 or bg > 180]
    patterns = []
    for (e1, t1), (e2, t2), (e3, t3) in zip(events, events[1:], events[2:]):
        if t1 == "hyper" and t2 == "hypo" and t3 == "hyper" and e2 - e1 <= 120 and e3 - e2 <= 120:
            patterns.append({"start": e1, "end": e3})
    return patterns

def detect_consistent_2am_spike_pattern_ad38(df):
    """
    ad_38: The CGM shows a glucose spike at 2 am for the past three nights. Is this consistent with my normal pattern?
    answer_generation_rule: Check whether 2 am spikes also appeared on other nights (beyond days 27–29).
    answer_instruction: Return a list of days that blood glucose exceeded 180 mg/dL around 2:00 am.
    answer_type: list of int
    metric: Accuracy
    """
    days = df['day'][(df["minute_of_day"] >= 60) & (df["minute_of_day"] <= 180) & (df["BG"] > 180)].unique()
    return sorted(int(x) for x in days.tolist())


def extract_post_lunch_faults_last_week_ad39(df):
    """
    ad_39: Is there any post-meal glucose response abnormal in the last week?
    answer_generation_rule: Extract last week's data. Intervals with fault labels during 12:00–14:00.
    answer_instruction: From days 24 to 30, return all time intervals between 12:00 and 14:00 where the blood glucose readings show abnormalities such as sudden spikes or drops, repeated values, or implausible insulin delivery patterns.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    valid_labels = [
        "negative_spike", "missing_signal", "repeated_reading", "positive_basal",
        "false_bolus", "negative_basal", "unknown_under", "min_basal",
        "unknown_stop", "max_reading", "min_reading", "negative_bias",
        "false_meal", "max_basal", "positive_spike"
    ]
    mask = (
        (df['day'] >= df['day'].max() -7) &
        df["minute_of_day"].between(720, 840) &
        df["faults_label"].isin(valid_labels)
    )
    return get_intervals_from_bool_mask(mask)

def extract_repeated_night_intervals_ad40(df):
    """
    ad_40: Did I have implausibly stable readings overnight?
    answer_generation_rule: Extract intervals between 12:00am–6:00am where readings are labeled as repeated_reading.
    answer_instruction: Return all time intervals between 00:00 and 06:00 where the blood glucose readings remained constant across multiple consecutive points, suggesting repeated sensor values.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df[df["minute_of_day"] < 360], ["repeated_reading"])

# def extract_best_last_week_ad41(df):
#     """
#     ad_41: On which day this week was my glucose pattern most ideal?
#     """

def extract_sensor_error_intervals_ad42(df):
    """
    ad_42: Were there any low glucose readings that might have been caused by a sensor error?
    answer_generation_rule: Extract periods with BG < 80 and faults_label indicating sensor issues (e.g., zero_reading, negative_bias, etc).
    answer_instruction: Return all time intervals where blood glucose readings were below 80 mg/dL and simultaneously exhibited abnormal patterns such as repeated values, zero readings, negative spikes, or consistently low readings, suggesting potential sensor errors.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    # df = df.iloc[:-1]
    sensor_labels = {"zero_reading", "negative_bias", "min_reading", "repeated_reading", "negative_spike"}
    mask = (df["BG"] < 80) & df["faults_label"].isin(sensor_labels)
    return get_intervals_by_label(df[mask], sensor_labels)


def extract_comm_loss_intervals_ad43(df):
    """
    ad_43: Did my CGM ever lose connection or stop communicating at any point?
    answer_generation_rule: Extract periods where the fault label is "missing_signal", indicating communication loss.
    answer_instruction: Return all time intervals where blood glucose data was missing or not recorded, suggesting possible communication loss between the CGM sensor and recorder.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["missing_signal"])

def extract_false_meal_ad44(df):
    """
    ad_44: Are there any likely wrong meal registrations?
    answer_generation_rule: Extract intervals where faults_label is "false_meal".
    answer_instruction: Return all intervals where a meal was registered even though the blood glucose was low and there was no physiological evidence of food intake, suggesting possible false meal entries.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["false_meal"])


def extract_replay_attack_ad45(df):
    """
    ad_45: Are there any likely false readings caused by replay attacks?
    answer_generation_rule: Extract intervals labeled as false_bolus, repeated_episode, or repeated_reading.
    answer_instruction: Return all time intervals where blood glucose data shows implausible repetition, reused meal patterns, or unexpected bolus events, which may indicate a replay attack.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["false_bolus", "repeated_reading", "repeated_episode"])


# def extract_workout_manipulation_ad46(df):
#     """
#     ad_46: Was there any dangerous operation happening with my controller during my usual workouts?
#     answer_generation_rule: Extract periods labeled as bias_basal, max_basal, or false_bolus during or after exercise.
#     answer_instruction: Return all time intervals during or immediately after physical activity periods where insulin delivery behavior appears abnormal, including excessive basal delivery or unexpected bolus injections.
#     answer_type: list of {"start": int, "end": int}
#     metric: Affinity F-score
#     """
#     # Need exercise info
#     return get_intervals_by_label(df, ["bias_basal", "max_basal", "false_bolus"])


def extract_bias_attack_ad47(df):
    """
    ad_47: Are there any readings that look suspicious, like they were affected by biased data tampering?
    answer_generation_rule: Extract intervals labeled as negative_bias or positive_bias.
    answer_instruction: Return all time intervals where blood glucose readings appear artificially shifted upward or downward for multiple consecutive points, suggesting possible data tampering.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["positive_bias", "negative_bias"])

def extract_insulin_pump_access_ad48(df):
    """
    ad_48: Has anyone accessed or controlled my insulin pump without my permission?
    answer_generation_rule: Extract periods labeled with false_bolus, bias_basal, max_basal, or min_basal.
    answer_instruction: Return all time intervals where insulin delivery shows abnormal patterns such as unauthorized bolus injection or abnormal basal rate changes, suggesting external access to the insulin pump.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["false_bolus", "bias_basal", "max_basal", "min_basal"])

def extract_incorrect_insulin_doses_ad49(df):
    """
    ad_49: Did I get any insulin doses that don't match what my controller normally does?
    answer_generation_rule: Extract intervals where faulty injection behavior is labeled (e.g., false_bolus, bias_basal, max_basal, min_basal).
    aanswer_instruction: Return all time intervals where insulin delivery behavior deviates from expected controller output, such as excessive, insufficient, or unexpected dosing.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["false_bolus", "bias_basal", "max_basal", "min_basal"])

def extract_saturation_spoofing_ad50(df):
    """
    ad_50: Are there any readings likely to be affected by saturation-based sensor spoofing attacks?
    answer_generation_rule: Extract periods labeled as max_reading.
    answer_instruction: Return all time intervals where blood glucose readings remain fixed at abnormally high values (e.g., 180–220 mg/dL), suggesting potential saturation-based spoofing of the sensor.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["max_reading"])

def detect_faulty_timestamps_ad51(df):
    """
    ad_51: Are there readings that might have been faked by false data injection attack?
    answer_generation_rule: Extract intervals labeled with known CGM reading-related injected faults.
    answer_instruction: Return all time intervals where blood glucose data shows sudden abnormal changes, repeated values, implausible insulin actions, or other known faulty patterns that may suggest tampering or spoofing.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    valid_labels = [
        "negative_spike", "missing_signal", "repeated_reading", "max_reading", "min_reading", "negative_bias",
        "positive_spike"
    ]
    return get_intervals_by_label(df, valid_labels)

def extract_harmful_pump_operations_ad52(df):
    """
    ad_52: Has the pump done anything that seems out of line or even harmful?
    answer_generation_rule: Extract intervals labeled with insulin delivery related faults.
    answer_instruction: Return all time intervals where basal insulin delivery exhibit abnormal patterns such as forced to its maximum or minimum rate, regardless of physiological demand, suggesting harmful pump behavior.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["max_basal", "min_basal", 'negative_basal', 'positive_spike', 'false_bolus'])

def extract_miscalibrated_sensor_ad53(df):
    """
    ad_53: Were there times when my sensor might have been miscalibrated?
    answer_generation_rule: Periods with faults_label == positive_bias or negative_bias.
    answer_instruction: Return all time intervals where blood glucose readings are consistently shifted up or down for multiple points, indicating potential sensor miscalibration.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["positive_bias", "negative_bias"])

def extract_pressure_sensor_attenuation_ad54(df):
    """
    ad_54: Is there any period that may indicate pressure-induced sensor attenuation?
    answer_generation_rule: Periods labeled as negative_spike.
    answer_instruction: Return all time intervals where blood glucose readings exhibit sudden sharp drops, which may reflect pressure-induced attenuation artifacts.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["negative_spike"])

# ad_55: Were there strange, sudden spikes in my readings that could’ve been caused by the exercise?
# ad_56: Were there periods when my sensor seemed less responsive or sensitive to changes in glucose?

def detect_negative_bias_periods_ad57(df):
    """
    ad_57: Is there any period that may indicate the lowered local glucose concentration?
    answer_generation_rule: Periods with faults_label == negative_bias and duration ≥ 6 hours (i.e., ≥ 72 data points for 5-minute sampling).
    answer_instruction: Return all time intervals of at least 6 hours in length (≥ 72 data points) where blood glucose readings show a sustained downward shift, suggesting lowered local glucose concentration.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    indices = df.index[df["faults_label"] == "negative_bias"].tolist()
    if not indices:
        return []
    intervals = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            if prev - start + 1 >= 72:  
                intervals.append({"start": start*5, "end": prev*5})
            start = prev = idx
    if prev - start + 1 >= 72:
        intervals.append({"start": start*5, "end": prev*5})
    return intervals

def detect_insulin_delivery_stop_ad58(df):
    """
    ad_58: Was there any point where it looked like insulin delivery may have stopped completely?
    answer_generation_rule: Periods with faults_label == min_basal or unknown_stop.
    answer_instruction: Return all time intervals where basal insulin delivery was either minimized or completely stopped, potentially indicating total delivery cessation.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["min_basal", "unknown_stop"])

def detect_insulin_delivery_issue_ad59(df):
    """
    ad_59: Was there any time when it looked like insulin was being delivered normally, but it actually wasn’t working or stopped?
    answer_generation_rule: Periods with faults_label == unknown_stop.
    answer_instruction: Return all time intervals where insulin delivery appeared normal from system logs but was actually stopped or failed internally.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["unknown_stop"])

def detect_insulin_under_delivery_ad60(df):
    """
    ad_60: Were there times when it looked like I wasn’t getting enough insulin?
    answer_generation_rule: Periods with data labelled as min_basal, negative_basal, or unknown_under.
    answer_instruction: Return all time intervals where actual insulin delivery was less than expected, including reduced basal rates, possible sensor bias, or incomplete absorption.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["min_basal", "negative_basal", "unknown_under"])

def detect_insulin_under_delivery_issue_ad61(df):
    """
    ad_61: Was there any time when it seemed like I was getting insulin normally, but the actual amount was less than what it was supposed to be?
    answer_generation_rule: Periods with faults_label == "unknown_under".
    answer_instruction: Return all time intervals where basal insulin was delivered at a lower rate than intended, despite the controller appearing to operate normally.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["unknown_under"])

def detect_empty_insulin_reservoir_ad62(df):
    """
    ad_62: Was there a time when it seemed like the insulin reservoir was empty and not giving me insulin as indicated?
    answer_generation_rule: Periods with faults_label == "unknown_stop".
    answer_instruction: Return all time intervals where insulin delivery stopped completely, which may indicate the insulin reservoir was depleted or disconnected.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["unknown_stop"])

def detect_blocked_inf_set_ad63(df):
    """
    ad_63: Could my infusion set have gotten blocked or kinked at any point?
    answer_generation_rule: Periods with faults_label == "unknown_stop".
    answer_instruction: Return all time intervals where insulin delivery ceased unexpectedly, possibly due to a blocked or kinked infusion set.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["unknown_stop"])

def detect_insulin_leakage_ad64(df):
    """
    ad_64: Was there any period when insulin might have leaked out or didn't get absorbed like it normally does?
    answer_generation_rule: Periods with faults_label == "unknown_under".
    answer_instruction: Return all time intervals where insulin was delivered but blood glucose did not respond as expected, suggesting possible insulin leakage or failed absorption.
    answer_type: list of {"start": int, "end": int}
    metric: Affinity F-score
    """
    return get_intervals_by_label(df, ["unknown_under"])