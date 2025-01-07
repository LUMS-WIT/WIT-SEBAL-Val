import numpy as np 

def clip_data(dates, values, min_val=0.12, max_val=0.90):
    """
    Clips the input dataset entries based on value thresholds, removing entries outside the specified range.

    Parameters
    ----------
    dates : numpy.array
        The array of dates corresponding to the values.
    values : numpy.array
        The array of data values to be filtered.
    min_val : float, optional
        The minimum allowable value. Entries below this will be removed. Default is 0.12.
    max_val : float, optional
        The maximum allowable value. Entries above this will be removed. Default is 0.90.

    Returns
    -------
    tuple
        Two numpy arrays: the filtered dates and their corresponding values.
    """
    filtered_indices = (values >= min_val) & (values <= max_val)
    return dates[filtered_indices], values[filtered_indices]

def remove_static_entries(data, window= 5):
    """
    Removes entries where the data remains static for five or more consecutive values.

    Parameters
    ----------
    dates : numpy.array
        Array of dates corresponding to the values.
    values : numpy.array
        Array of data values.

    Returns
    -------
    tuple
        Two numpy arrays: the filtered dates and their corresponding values after removing static entries.
    """
    dates, values = data
    values = np.array(values)
    if len(values) < window:
        return dates, values  # Not enough data to have static entries

    # Create a boolean array where True indicates the value is the same as the next
    static_mask = np.r_[True, values[:-1] == values[1:], True]

    # Create groups of consecutive same values
    counts = np.diff(np.where(static_mask)[0])

    # Find the indices of groups with 5 or more consecutive same values
    remove_indices = np.repeat(counts >= window, counts)
    
    # Apply the mask to filter out static entries
    filtered_dates = dates[~remove_indices]
    filtered_values = values[~remove_indices]
    return filtered_dates, filtered_values

def sms_calibrations(data):
    """
    Applies clipping to the data values within the specified limits.

    Parameters
    ----------
    data : tuple
        A tuple containing an array of dates and an array of corresponding values.

    Returns
    -------
    tuple
        A tuple containing the original dates and the clipped values.
    """
    dates, values = data
    clipped_values_ = clip_data(np.array(dates), np.array(values)) # Ensure to convert list to numpy array if not already
    # clipped_values = remove_static_entries(clipped_values_)
    dates, values = clipped_values_
    
    return (dates, values)
