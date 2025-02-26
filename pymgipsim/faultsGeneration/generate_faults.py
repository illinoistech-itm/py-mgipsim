import numpy as np
import pandas as pd

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



def generate_faults(faults_file, simulation_days, simulation_start_time):
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
        time_delta = row['Start_Time'] - simulation_start_time
        start_minute = int(time_delta.total_seconds() / 60)
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

    return fault_input


if __name__ == '__main__':
    simulation_start_time = pd.Timestamp('2023-01-01 00:00:00')
    print("Simulation start time: ", simulation_start_time)
    print("--- Fault Label to Integer ID Mapping ---")
    for label, num in faults_label_dict.items():
        print(f"'{label}': {num}")
    print("-" * 35)

    faults_spec_file = r"pymgipsim/faultsGeneration/faults_specification.csv"
    faults_input = generate_faults(faults_spec_file, 30, simulation_start_time)





