import random
import copy
from datetime import timedelta


def randomize_events(input_settings, target, random_way=None, intensity=0.1, random_state=100):
    '''
    Input parameters:
        settings_file: scenario generated with default settings
        intensity: the percentage of changed events
        target: one or several targets in input settings, e.g., ['meal_carb', 'meal_start_time']
        random_way: default is None, which introduces all randoms ways. Or indicating random ways, e.g., ['heavy', 'delayed']

    Function:
        Randomly change the input settings (list: [patient_number, events_number]):
            meal_carb: settings_file.inputs.meal_carb.magnitude
            meal_start_time: settings_file.inputs.meal_carb.start_time
            snack_carb: settings_file.inputs.snack_carb.magnitude
            snack_start_time: settings_file.inputs.snack_carb.start_time
            cycling_power: settings_file.inputs.cycling_power.magnitude
            cycling_start_time: settings_file.inputs.cycling_power.start_time
            cycling_duration: settings_file.inputs.cycling_power.duration
            running_speed: settings_file.inputs.running_speed.magnitude
            running_start_time: settings_file.inputs.running_speed.start_time
            running_duration: settings_file.inputs.running_speed.duration


        by following ways:
            heavy: Increase 10-40% of magnitude or duration
            light: Decrease 10-40% of magnitude or duration
            early: 1-2 hour before original start_time
            delayed: 1-2 hour after original start_time
            skipped: Set 0 of magnitude

        of randomly selected [total_events*intensity] events

    Output:
        Updated settings_file
    '''

    random.seed(random_state)

    # Define available randomization methods
    change_methods = {
        "heavy": lambda val: val * round(random.uniform(1.1, 1.5), 2),
        "light": lambda val: val * round(random.uniform(0.4, 0.9), 2),
        "skipped": lambda val: 0,                        # Set magnitude to 0
        "early": lambda val: val - random.randint(60, 120),    # 1–2 hours earlier (in minutes)
        "delayed": lambda val: val + random.randint(60, 120),  # 1–2 hours later (in minutes)
    }

    # Allowed change types per attribute kind
    allowed_changes = {
        "magnitude": ["heavy", "light", "skipped"],
        "start_time": ["early", "delayed"],
        "duration": ["heavy", "light"]
    }

    if random_way is None:
        random_way = list(change_methods.keys())

    # Deep copy to avoid modifying the original
    updated_settings = copy.deepcopy(input_settings)

    # Map target names to actual object paths
    attr_map = {
        "meal_carb": ("meal_carb", "magnitude"),
        "meal_start_time": ("meal_carb", "start_time"),
        "snack_carb": ("snack_carb", "magnitude"),
        "snack_start_time": ("snack_carb", "start_time"),
        "cycling_power": ("cycling_power", "magnitude"),
        "cycling_start_time": ("cycling_power", "start_time"),
        "cycling_duration": ("cycling_power", "duration"),
        "running_speed": ("running_speed", "magnitude"),
        "running_start_time": ("running_speed", "start_time"),
        "running_duration": ("running_speed", "duration"),
    }

    # Gather all (object_ref, attr_name, patient_idx, event_idx) tuples
    all_events = []
    for event_name in target:
        attr_path = attr_map[event_name]
        obj = updated_settings
        for attr in attr_path[:-1]:
            obj = getattr(obj, attr)
        values_per_patient = getattr(obj, attr_path[-1])  # list of lists

        for patient_idx, patient_values in enumerate(values_per_patient):
            for event_idx in range(len(patient_values)):
                all_events.append((obj, attr_path[-1], patient_idx, event_idx))

    # Randomly choose events to modify
    num_changes = int(len(all_events) * intensity)
    events_to_change = random.sample(all_events, num_changes)

    # Apply changes
    for obj, attr_name, patient_idx, event_idx in events_to_change:
        patient_values = getattr(obj, attr_name)[patient_idx]
        current_val = patient_values[event_idx]

        # Choose valid change types for this kind
        valid_changes = allowed_changes[attr_name]
        if random_way:
            valid_changes = [c for c in valid_changes if c in random_way]
        if not valid_changes:
            continue  # no valid change types

        method = random.choice(valid_changes)
        new_val = change_methods[method](current_val)
        patient_values[event_idx] = new_val

    return updated_settings