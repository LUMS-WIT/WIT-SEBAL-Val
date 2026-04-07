import numpy as np
import warnings
import datetime

def mean_std(x, y):
    """
    scales the input datasets so that they have the same mean
    and standard deviation afterwards

    Parameters
    ----------
    src : numpy.array
        input dataset which will be scaled
    ref : numpy.array
        src will be scaled to this dataset

    Returns
    -------
    scaled dataset : numpy.array
        dataset src with same mean and standard deviation as ref
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Input arrays must each contain at least two elements.")
    return ((x - np.mean(x)) / np.std(x)) * np.std(y) + np.mean(y)

def scaling(test, ref):
    """
    Scales the test dataset to have the same mean and standard deviation as the reference dataset.
    The function filters test data to match the temporal extent of reference data and then applies
    scaling if there are enough data points. It warns if the data is insufficient for scaling.this 
    makes sures that data is only used if values for X and y are available, so that seasonal 
    patterns in missing values in one of them do not lead to distortion.

    Parameters
    ----------
    test : tuple
        A tuple containing two elements: an array of dates and an array of corresponding values 
        for the test dataset.
    ref : tuple
        A tuple containing two elements: an array of dates and an array of corresponding values 
        for the reference dataset.

    Returns
    -------
    tuple
        A tuple containing the potentially scaled test dataset (dates and values) and the unchanged 
        reference dataset.
    """
    # Unpack the reference and test data
    test_dates, x = test
    ref_dates, y = ref

    # Find the minimum and maximum dates in the reference data
    min_ref_date = min(ref_dates)
    max_ref_date = max(ref_dates)

    # Filter the test data to only include dates within the ref_data date range
    filtered_test_dates = []
    filtered_x = []
    for date, value in zip(test_dates, x):
        if min_ref_date <= date <= max_ref_date:
            filtered_test_dates.append(date)
            filtered_x.append(value)

    # Extend the range to include one before and after the filtered data
    start_index = max(test_dates.index(filtered_test_dates[0]) - 1, 0) if filtered_test_dates else 0
    end_index = min(test_dates.index(filtered_test_dates[-1]) + 1, len(test_dates) - 1) if filtered_test_dates else len(test_dates) - 1

    extended_test_dates = test_dates[start_index:end_index+1]
    extended_x = x[start_index:end_index+1]

    # Apply mean_std scaling if there are enough data points
    if len(filtered_x) >= 2:
        scaled_test_values = mean_std(np.array(extended_x), np.array(y))
        scaled_test = (extended_test_dates, scaled_test_values)
    else:
        warnings.warn("Not enough data points to apply scaling.")
        scaled_test = test

    return scaled_test, ref

def temporal_matching(test_data, ref_data, search_range=0):
    """
    Finds overlapping dates by checking a range of days around each reference date to find
    the closest matching test date with a valid (non-NaN) value.
    Note that valid overlap count cannot exceed than the number of points in reference data

    Parameters:
    - ref_data: Tuple containing two lists (dates and values) for the reference dataset.
    - test_data: Tuple containing two lists (dates and values) for the test dataset.
    - search_until_range: Number of days to check around each reference date.

    Returns:
    - overlapping_ref_dates: List of reference dates that have overlaps.
    - overlapping_ref_values: List of values from the reference dataset for the overlapping dates.
    - overlapping_test_values: List of closest values from the test dataset corresponding to the reference dates.
    - valid_overlap_count: Count of valid overlapping points.
    """
    ref_dates, ref_values = ref_data
    test_dates, test_values = test_data

    # Convert datetime.datetime to datetime.date to ignore time of day
    ref_data_dict = {date.date(): value for date, value in zip(ref_dates, ref_values)}
    test_data_dict = {date.date(): value for date, value in zip(test_dates, test_values)}

    overlapping_ref_dates = []
    overlapping_ref_values = []
    overlapping_test_values = []

    for ref_date, ref_value in ref_data_dict.items():
        closest_value = None
        closest_distance = search_range + 1  # Initialize to more than the search range

        # Check dates within the range around the reference date
        for offset in range(-search_range, search_range + 1):
            test_date = ref_date + datetime.timedelta(days=offset)
            if test_date in test_data_dict and not np.isnan(test_data_dict[test_date]):
                distance = abs(offset)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_value = test_data_dict[test_date]

        # If a valid closest value is found within the range
        if closest_value is not None:
            overlapping_ref_dates.append(ref_date)
            overlapping_ref_values.append(ref_value)
            overlapping_test_values.append(closest_value)

    valid_overlap_count = len(overlapping_ref_dates)

    return overlapping_ref_dates, overlapping_ref_values, overlapping_test_values, valid_overlap_count


def temporal_matching_windowed(model_data, sensor_data, half_window=0, min_valid=1, agg_func=np.nanmean):
    """
    Match each MODEL/SEBAL date exactly once with a windowed SENSOR summary.

    Parameters
    ----------
    model_data : tuple
        (model_dates, model_values) e.g. SEBAL dates and values
    sensor_data : tuple
        (sensor_dates, sensor_values) e.g. WIT sensor dates and values
    half_window : int
        0 -> same-day only
        1 -> centered 3-day window
        2 -> centered 5-day window
        3 -> centered 7-day window
    min_valid : int
        Minimum number of valid sensor observations required inside the window.
    agg_func : callable
        Aggregation function for valid sensor values, default np.nanmean

    Returns
    -------
    matched_model_dates : list
        Dates of SEBAL observations that found enough valid sensor support
    aggregated_sensor_values : list
        Window-aggregated sensor values
    matched_model_values : list
        Corresponding SEBAL values
    valid_overlap_count : int
        Number of matched SEBAL dates
    n_sensor_used : list
        Number of valid sensor values used in each aggregated window
    used_sensor_dates : list
        List of actual sensor dates used for each SEBAL date
    """
    model_dates, model_values = model_data
    sensor_dates, sensor_values = sensor_data

    # Convert to daily dictionaries
    sensor_dict = {
        d.date(): float(v)
        for d, v in zip(sensor_dates, sensor_values)
        if not np.isnan(v)
    }

    matched_model_dates = []
    aggregated_sensor_values = []
    matched_model_values = []
    n_sensor_used = []
    used_sensor_dates = []

    # Sort model dates so output is ordered
    for model_dt, model_val in sorted(zip(model_dates, model_values), key=lambda x: x[0]):
        if np.isnan(model_val):
            continue

        center_date = model_dt.date()
        window_values = []
        window_dates_used = []

        for offset in range(-half_window, half_window + 1):
            candidate_date = center_date + datetime.timedelta(days=offset)

            if candidate_date in sensor_dict:
                v = sensor_dict[candidate_date]
                if not np.isnan(v):
                    window_values.append(v)
                    window_dates_used.append(candidate_date)

        # Only keep if enough valid sensor values exist
        if len(window_values) >= min_valid:
            agg_value = float(agg_func(window_values))

            matched_model_dates.append(center_date)
            aggregated_sensor_values.append(agg_value)
            matched_model_values.append(float(model_val))
            n_sensor_used.append(len(window_values))
            used_sensor_dates.append(window_dates_used)

    valid_overlap_count = len(matched_model_dates)

    return (
        matched_model_dates,
        aggregated_sensor_values,
        matched_model_values,
        valid_overlap_count,
        n_sensor_used,
        used_sensor_dates,
    )
