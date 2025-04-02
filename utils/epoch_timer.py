def epoch_time(start_time, end_time):
    """
    Args:
        start_time: The start time in seconds since the epoch.
        end_time: The end time in seconds since the epoch.

    Returns:
        Tuple containing elapsed minutes and elapsed seconds between start_time and end_time.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
