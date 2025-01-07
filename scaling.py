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
