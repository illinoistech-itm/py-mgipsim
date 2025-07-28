import json
import numpy as np
from datetime import datetime
from datetime import timedelta
from preprocess_data import preprocess_data as prepdata
from preprocess_data import format_time_info
import fire
import os
import re
import random


def generate_questions_and_answers(patient_data):
    """Generate questions and calculate ground truth answers."""

    bg_values = patient_data["bg_mgdl"]
    insulin_events = patient_data["insulin_events"]
    
    # Blood glucose related calculations per day
    daily_bg = {}
    samples_per_day = 24 * 12  # 288 samples per day (every 5 minutes)
    num_days = len(bg_values) // samples_per_day


    # overall statistics
    total_insulin = sum(event["dosage"] for event in insulin_events)
    
    largest_bolus = max((event for event in insulin_events if event["insulin_type"] == "bolus_insulin"), key=lambda x: x["dosage"])
    largest_bolus_time = f"largest_bolus['time_str']" 
    largest_bolus_amount = largest_bolus["dosage"]
    basal_events = [event for event in insulin_events if event["insulin_type"] == "basal_insulin"]


    # Insulin events for each day
    for day in range(1, num_days + 1):
        start_idx = (day - 1) * samples_per_day
        end_idx = min(day * samples_per_day, len(bg_values))
        day_bg_values = bg_values[start_idx:end_idx]
        morning_values = day_bg_values[72:144]     # 6:00–12:00
        afternoon_values = day_bg_values[144:216]   # 12:00–18:00

        if len(day_bg_values) == 0:
            continue

        day_insulin_events = [
            event for event in insulin_events if event["day"] == day  
        ]
        total_insulin = sum(event["dosage"] for event in day_insulin_events)

        boluses = [e for e in day_insulin_events if e["insulin_type"] == "bolus_insulin"]
        largest_bolus_event = max(boluses, key=lambda x: x["dosage"], default=None)
        largest_bolus_amount = largest_bolus_event["dosage"] if largest_bolus_event else None
        largest_bolus_time = largest_bolus_event["time_str"] if largest_bolus_event else None

        # Store daily statistics
        daily_bg[f"day{day}"] = {
            "bg": day_bg_values,
            "total_insulin": round(total_insulin, 2),
            "largest_bolus_amount": largest_bolus_amount,
            "largest_bolus_time": largest_bolus_time,
        }

    # Generate questions and answers
    questions_and_answers = []

    # descriptive
    questions_and_answers.append({
        "question": "What was the patient's total daily insulin dose?",
        "answer": f"{total_insulin:.2f} units",
        "answer_generation_rule": "Sum all insulin amounts from insulin events.",
        "answer_instruction": "Return the sum of all insulin doses units across the day, rounded to two decimal places.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "34.00 units"
    })

    questions_and_answers.append({
        "question_text": "When did the patient receive their largest insulin bolus?",
        "answer": largest_bolus_time, 
        "answer_generation_rule": "Find the insulin event with the maximum insulin amount.",
        "answer_instruction": "Return a tuple representing the time of the largest bolus as <HH:MM>.",
        "answer_type": "time_str",
        "metric": "Accuracy",
        "example_answer": "07:45"
    })

    if len(basal_events) > 1:
        questions_and_answers.append({
            "question_text": "How did basal rates change throughout the day on day x?",
            "answer": {
                "num_adjustments": len(basal_events),
                "avg_dosage": round(np.mean([e["dosage"] for e in basal_events]), 4)
            },
            "answer_generation_rule": "Analyze insulin events below threshold (assumed to be basal) for frequency and amount.",
            "answer_instruction": "Return a dictionary with the number of basal rate adjustments and the average dosage per adjustment rounded to four decimal places, using keys: 'num_adjustments' and 'avg_dosage'.",
            "answer_type": "dict",
            "metric": "dict of MAE",
            "example_answer": {
                "num_adjustments": 18,
                "avg_dosage": 0.0200
            }
        })
    
    # memory/temporal
    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    day_name = f"day {random_day}"

    questions_and_answers.append({
        "question": f"What was the patient's total daily insulin dose on {day_name}?",
        "answer": f"{daily_bg[day_key]['total_insulin']:.2f} units",
        "answer_generation_rule": f"Sum all basal and bolus insulin amounts recorded throughout {day_name}.",
        "answer_instruction": f"Return the total insulin dose on {day_name}, rounded to two decimal places, followed by 'units'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "34.00 units"
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    day_name = f"day {random_day}"

    questions_and_answers.append({
        "question": f"When did the patient receive their largest insulin bolus on {day_name}?",
        "answer": f"{daily_bg[day_key]['largest_bolus_time']}",  
        "answer_generation_rule": f"Find the insulin bolus event with the highest insulin amount on {day_name} and return its time.",
        "answer_instruction": f"Return the time of the largest bolus on {day_name} in the format <HH:MM>.",
        "answer_type": "time_str",
        "metric": "Accuracy",
        "example_answer": "07:45"
    })

    weekday_insulin = 0.0
    weekend_insulin = 0.0

    for day_key, bg in daily_bg.items():
        day_num = int(day_key.replace("day", ""))  # 1=Monday, ..., 7=Sunday
        insulin = bg.get("insulin_total", 0.0)

        if day_num in [1, 2, 3, 4, 5]:
            weekday_insulin += insulin
        elif day_num in [6, 7]:
            weekend_insulin += insulin
    
    avg_weekday_insulin = round(weekday_insulin / 5, 2)
    avg_weekend_insulin = round(weekend_insulin / 2, 2)

    questions_and_answers.append({
        "question": "Does the patient use more insulin on weekends in the the first week?",
        "answer": f"{"Yes" if avg_weekend_insulin > avg_weekday_insulin else "No"}",
        "answer_generation_rule": (
            "Calculate the average insulin doses for weekend (day 6 and day 7) of the the week. "
            "Compare with the average daily insulin use on weekdays (day 1 to 5) this week. "
            "If the weekend average is greater than weekday average, return 'Yes'; otherwise, return 'No'."
        ),
        "answer_instruction": (
            "Return 'Yes' if the average of insulin use on weekends is higher than the average insulin use on weekdays in this week; "
            "otherwise, return 'no'."
        ),
        "answer_type": "Yes or No",
        "metric": "Accuracy",
        "example_answer": "Yes"
    })

    return questions_and_answers


def process_jsonl_file(input_file, output_file, include_patient_data=True):
    """
    Process JSONL file, generate questions and answers, write to output file.
    
    Args:
        input_file: Path to input JSONL file with glucose data
        output_file: Path to output JSONL file for questions and answers
        include_patient_data: If True, include the original patient data in the output
    """
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        line_count = 0
        processed_count = 0
        
        for line in f_in:
            line_count += 1
            try:
                patient_data = json.loads(line.strip())
                
                results = {
                    "patient_id": patient_data["patient_id"],
                    "qa_pairs": generate_questions_and_answers(patient_data)
                }
                for i, qa in enumerate(results["qa_pairs"]):
                    qa["question_id"] = f"pm_insulin_{i}"

                if include_patient_data:
                    results["input_context"] = patient_data
                
                f_out.write(json.dumps(results) + '\n')
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} records...")
                
            except Exception as e:
                print(f"Error processing line {line_count}: {e}")
                import traceback
                traceback.print_exc()  # Added for better debugging
        
        print(f"Processing complete. Successfully processed {processed_count} of {line_count} records.")

def main(input_file=None, 
         output_file=None,
         include_patient_data=True):
    """
    Generate glucose-related questions and answers from patient data.
    
    Args:
        input_file: Path to input JSONL file containing glucose trace data
        output_file: Path to output JSONL file for questions and answers
        include_patient_data: Whether to include original patient data in output (default: True)
    """
    process_jsonl_file(input_file, output_file, include_patient_data)



if __name__ == "__main__":
    day = 30
    controller = "openloop"
    scenario_name = "morning_runner"
    base_path = f"./SimulationData/{scenario_name}_{controller}_insulin"
    output_path = "./QA_pairs"
    os.makedirs(output_path, exist_ok=True)

    patient_id = "morning_runner_1"
    input_file = os.path.join(base_path, f"{patient_id}_simulation_data.jsonl")
    output_file = os.path.join(output_path, f"{patient_id}_questions_answers_{controller}_insulin.jsonl")
    main(input_file, output_file, include_patient_data=True)