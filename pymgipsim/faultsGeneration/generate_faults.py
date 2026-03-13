import numpy as np
import pandas as pd
import random

faults_label_dict = {
    'max_basal': 1, 'min_basal': 2, 'positive_basal': 3, 'negative_basal': 4, 'unknown_stop': 5,
    'unknown_under': 6, 'missing_signal': 7, 'positive_spike': 8, 'negative_spike': 9, 'negative_bias': 10,
    'positive_bias': 11, 'min_reading': 12, 'max_reading': 13, 'repeated_reading': 14, 'false_meal': 15,
    'false_bolus': 16, 'repeated episode': 17
}

faults_id_dict = {
    1: 'max_basal', 2: 'min_basal', 3: 'positive_basal', 4: 'negative_basal', 5: 'unknown_stop',
    6: 'unknown_under', 7: 'missing_signal', 8: 'positive_spike', 9: 'negative_spike', 10: 'negative_bias',
    11: 'positive_bias', 12: 'min_reading', 13: 'max_reading', 14: 'repeated_reading', 15: 'false_meal',
    16: 'false_bolus', 17: 'repeated episode'
}



def generate_faults_from_file(faults_file, simulation_days):
    """
    Generates a 1D NumPy array representing a timeline of faults.

    Each element in the array corresponds to a minute in the simulation.
    The value of the element is an integer ID for the fault occurring at
    that minute, or 0 if no fault is present.

    Args:
        faults_file: CSV file with columns ['Start_Time', 'Period', 'Data Label', 'Description'].
                           'Start_Time': Fault start time. Should be a datetime object.
                           'Period': Length of fault injection. Count in minutes.
                           'Data Label': Fault category.
                           'Description': Explain the attack or malfunction simulated here.
        args.number_of_days (int): The total number of days for the simulation timeline.
        simulation_start_time: align simulation time with real-world time
        sampling_time (int): The sampling interval in minutes. The final output will be
                     sampled at this rate. Defaults to 1.
    Returns:
            - np.ndarray: The fault timeline array of shape (1, simulation_days * 1440).
    """
    df = pd.read_csv(faults_file, parse_dates=['Start_Time'])
    simulation_len = simulation_days * 24 * 60
    fault_input = np.zeros(simulation_len, dtype=int)

    for index, row in df.iterrows():
        # Skip rows that don't have a valid start time (like the 'No Fault' days)
        if pd.isna(row['Start_Time']):
            continue

        fault_id = faults_label_dict.get(row['Data Label'])
        if fault_id is None:
            continue # Skip if the label is not in our map

        # Calculate the start minute index for this event
        # time_delta = row['Start_Time'] - simulation_start_time
        # start_minute = int(time_delta.total_seconds() / 60)
        start_minute = int(row['Start_Time'])
        # Get the duration of the fault in minutes
        duration_minutes = int(row['Period'])
        # Calculate the end minute (exclusive)
        end_minute = start_minute + duration_minutes

        # Ensure we don't write past the end of the array
        if start_minute >= simulation_len:
            print(f"Warning: Fault '{row['Data Label']}' at {row['Start_Time']} is outside the simulation period.")
            continue

        # "Paint" the fault ID onto the timeline for its duration
        fault_input[start_minute:end_minute] = fault_id

    # # Sampling: downsample every `sampling_time` minutes using max value in window
    # sampled_length = simulation_len // sampling_time
    # fault_input = np.array([
    #     np.max(fault_input[i * sampling_time:(i + 1) * sampling_time])
    #     for i in range(sampled_length)
    # ])

    return fault_input



def generate_random_faults(simulation_days, intensity=0.1, random_state=100, faulty_type=None):
    """
    Randomly inject non-overlapping faults into a timeline of given duration.

    Args:
        simulation_days (int): Number of simulation days (1 day = 1440 minutes).
        intensity (float): Ratio of total simulation time that should be fault-injected.
        random_state (int): Seed for reproducibility.
    Returns:
        np.ndarray: Array of shape (simulation_days * 1440,), with fault IDs injected.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    simulation_len = simulation_days * 24 * 60
    fault_input = np.zeros(simulation_len, dtype=int)
    used_indices = set()

    # Special-case fault types
    one_point_faults = {8, 9, 15, 16}
    repeated_fault = 17

    # Estimate total number of fault minutes based on intensity
    total_fault_minutes = int(simulation_len * intensity)

    def is_valid_range(start, duration):
        return all(i not in used_indices for i in range(start, start + duration)) and start + duration <= simulation_len

    def mark_range(start, duration, fault_id):
        for i in range(start, start + duration):
            fault_input[i] = fault_id
            used_indices.add(i)

    target_faults_id_dict = {k: v for k, v in faults_id_dict.items() if v in faulty_type}

    for fault_id in target_faults_id_dict:
        injected = False
        tries = 0
        while not injected and tries < 1000:
            tries += 1
            if fault_id in one_point_faults:
                start = random.randint(0, simulation_len - 1)
                if is_valid_range(start, 1):
                    mark_range(start, 1, fault_id)
                    total_fault_minutes -= 1
                    injected = True
            elif fault_id == repeated_fault:
                duration = random.randint(120, 240)  # 2–4 hours
                start = random.randint(0, simulation_len - duration)
                if is_valid_range(start, duration):
                    mark_range(start, duration, fault_id)
                    total_fault_minutes -= duration
                    injected = True
            else:
                duration = random.randint(15, 120)  # 15 min – 2 hours
                start = random.randint(0, simulation_len - duration)
                if is_valid_range(start, duration):
                    mark_range(start, duration, fault_id)
                    total_fault_minutes -= duration
                    injected = True

    # If there's remaining quota for faults, randomly inject more from any fault
    all_fault_ids = list(target_faults_id_dict.keys())
    while total_fault_minutes > 0:
        fault_id = random.choice(all_fault_ids)
        if fault_id in one_point_faults:
            duration = 1
        elif fault_id == repeated_fault:
            duration = random.randint(120, 300)
        else:
            duration = random.randint(15, 120)

        if duration > total_fault_minutes:
            break

        start = random.randint(0, simulation_len - duration)
        if is_valid_range(start, duration):
            mark_range(start, duration, fault_id)
            total_fault_minutes -= duration

    return fault_input






if __name__ == '__main__':
    simulation_start_time = pd.Timestamp('2023-01-01 00:00:00')
    print("Simulation start time: ", simulation_start_time)
    print("--- Fault Label to Integer ID Mapping ---")
    for label, num in faults_label_dict.items():
        print(f"'{label}': {num}")
    print("-" * 35)

    faults_spec_file = r"pymgipsim/faultsGeneration/faults_specification.csv"
    faults_input = generate_faults_from_file(faults_spec_file, 30, simulation_start_time)





