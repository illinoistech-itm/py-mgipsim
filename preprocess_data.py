import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta

def resample_input_field(data, field_name, sampling_minutes=5): 
    field = data["inputs"][field_name]
    magnitudes = field["magnitude"]
    start_times = field["start_time"]

    resampled = {
        "magnitude": [],
        "start_time": [],
        "duration": []
    }

    minutes_per_day = 1440
    step = sampling_minutes

    for person_index in range(len(magnitudes)):
        mag = np.array(magnitudes[person_index])
        st = np.array(start_times[person_index])

        total_minutes = len(mag)

        if total_minutes % minutes_per_day != 0:
            print(f"Warning: Person {person_index} has incomplete days — skipping.")
            continue

        days = total_minutes // minutes_per_day
        mag = mag.reshape((days, minutes_per_day))[:, ::step]
        st = st.reshape((days, minutes_per_day))[:, ::step]

        resampled["magnitude"].append(mag.flatten().tolist())
        resampled["start_time"].append(st.flatten().tolist())

    return resampled

def format_time_info(mins):
    mins = float(mins)
    day = int(mins // 1440) + 1  # ← day 1 to 14, no weekly reset
    time_of_day = mins % 1440
    hours = int(time_of_day // 60)
    minutes = int(time_of_day % 60)
    time_str = f"{hours:02d}:{minutes:02d}" 
    return day, time_str


def extract_carb_events(data, i):
    events = []

    # Meals: [breakfast, lunch, dinner]
    if "meal_carb" in data["inputs"]:
        mags = data["inputs"]["meal_carb"]["magnitude"][i]
        times = data["inputs"]["meal_carb"]["start_time"][i]
        meal_types = ["breakfast", "lunch", "dinner"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t,
                "day": day,
                "time_str": time_str,
                "carbs": carbs,
                "meal_type": meal_types[idx % 3]
            })

    # Snacks: [morning_snack, afternoon_snack]
    if "snack_carb" in data["inputs"]:
        mags = data["inputs"]["snack_carb"]["magnitude"][i]
        times = data["inputs"]["snack_carb"]["start_time"][i]
        snack_types = ["morning_snack", "afternoon_snack"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t,
                "day": day,
                "time_str": time_str,
                "carbs": carbs,
                "meal_type": snack_types[idx % 2]
            })

    events.sort(key=lambda x: x["time"])
    return events

def extract_insulin_events(data, i):
    events = []
    for source in ["basal_insulin", "bolus_insulin"]:
        if source not in data["inputs"]:
            continue
        magnitudes = data["inputs"][source]["magnitude"][i]
        start_times = data["inputs"][source]["start_time"][i]
        for dos, t in zip(magnitudes, start_times):
            if dos == 0:
                continue  # Skip zero-dosage entries
            day, time_str = format_time_info(t)
            events.append({
                "time": t,
                "day": day,
                "time_str": time_str,
                "dosage": dos,
                "insulin_type": source,
            })
    events.sort(key=lambda x: x["time"])
    return events

def extract_exercise_events_combined(data, i):
    events = []
    for source, label in [("running_speed", "running"), ("cycling_power", "cycling")]:
        if source not in data["inputs"]:
            continue
        magnitudes = data["inputs"][source]["magnitude"][i]
        start_times = data["inputs"][source]["start_time"][i]
        durations = data["inputs"][source]["duration"][i]
        for mag, t, d in zip(magnitudes, start_times, durations):
            if mag == 0:
                continue  # skip zero magnitude events
            day, time_str = format_time_info(t)
            events.append({
                "time": t,
                "day": day,
                "time_str": time_str,
                "duration": d,
                "magnitude": mag,
                "exercise_type": label,
            })
    events.sort(key=lambda x: x["time"])
    return events


def preprocess_data(simulation_path, bg_path, output_path, num_people, scenario_name):
    with open(simulation_path, "r") as f:
        data = json.load(f)

    #data["inputs"]["heart_rate"] = resample_input_field(data, field_name="heart_rate", sampling_minutes=5) # resample heart_rate

    for i in range(num_people): 
        sheet_name = f"Patient_{i}"
        simulation_data = {}

        simulation_data["patient_id"] = f"{scenario_name}_{i}"

        # Carbs
        simulation_data["carb_events"] = extract_carb_events(data, i)

        # Insulin
        insulin_events = extract_insulin_events(data, i)
        if insulin_events:
            simulation_data["insulin_events"] = insulin_events # skip empty insulin events

        # Exercise
        exercise_events = extract_exercise_events_combined(data, i)
        if exercise_events:
            simulation_data["exercise_events"] = exercise_events

        # Heart rate as list
        #simulation_data["heart_rate"] = data["inputs"]["heart_rate"]["magnitude"][i]

        # Blood glucose
        try:
            df = pd.read_excel(bg_path, sheet_name=sheet_name)
            df["BG"] = df["IG (mmol/L)"] * 18
            print(f"Loaded data for {sheet_name}: {df.shape}")

            simulation_data["bg_mgdl"] = df["BG"].tolist()
            # simulation_data["bg_time"] = df["start_time"].tolist() bg_time = df["BG"].index * 5

        except Exception as e:
            print(f"Error loading {sheet_name}: {e}")
            simulation_data["bg_mgdl"] = []


        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f"{scenario_name}_{i}_simulation_data.jsonl"), "w") as f:
            f.write(json.dumps(simulation_data) + "\n")


if __name__ == "__main__":
    FOLDER_NAME = "morning_runner_openloop_insulin"
    BASE_PATH = os.path.join("SimulationResults", FOLDER_NAME)
    SIMULATION_PATH = os.path.join(BASE_PATH, "simulation_settings.json")
    BG_PATH = os.path.join(BASE_PATH, "model_state_results.xlsx")
    OUTPUT_PATH = os.path.join("SimulationData", FOLDER_NAME)
    NUM_PEOPLE = 2

    preprocess_data(
        simulation_path=SIMULATION_PATH,
        bg_path=BG_PATH,
        output_path=OUTPUT_PATH,
        num_people=NUM_PEOPLE,
        scenario_name="morning_runner"
    )

