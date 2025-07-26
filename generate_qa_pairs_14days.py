import json
import numpy as np
from datetime import datetime
from datetime import timedelta
from preprocess_data import preprocess_data as prepdata
from preprocess_data import format_time_info
import fire
import os
import re


def generate_questions_and_answers(patient_data):
    """Generate questions and calculate ground truth answers."""

    #Calculate weekly metrics
    bg_values = patient_data["bg_mgdl"]  # list of bg values every 5 mins
    readings_per_day = 288
    readings_per_week = readings_per_day * 7

    week1_bg = bg_values[:readings_per_week]
    week2_bg = bg_values[readings_per_week:readings_per_week * 2]

    def get_metrics(week_bg):
        week_array = np.array(week_bg)
        mean = np.mean(week_array)
        std = np.std(week_array)
        cv = round((std / mean) * 100, 2) if mean > 0 else None

        tir = np.sum((week_array >= 70) & (week_array <= 180)) / len(week_array) * 100
        hypoglycemia_time_min = np.sum(week_array < 70) * 5
        hyperglycemia_time_min = np.sum(week_array > 180) * 5

        return {
            "mean": round(mean, 1),
            "std": round(std, 1),
            "cv": cv,
            "tir_percent": round(tir, 1),
            "hypoglycemia_minutes": int(hypoglycemia_time_min),
            "hyperglycemia_minutes": int(hyperglycemia_time_min),
        }
    
    week1_stats = get_metrics(week1_bg)
    week2_stats = get_metrics(week2_bg)

    # Generate questions and answers
    questions_and_answers = []
    
    questions_and_answers.append({
        "question": "Was the time spent in hypoglycemia lower this week compared to last week?",
        "answer": "yes" if week2_stats["hypoglycemia_minutes"] < week1_stats["hypoglycemia_minutes"] else "no",
        "answer_generation_rule": "Sum minutes with glucose < 70 mg/dL in each week and compare.",
        "answer_instruction": "Return 'yes' if time in hypoglycemia (<70 mg/dL) is lower this week than last, otherwise return 'no'.",
        "answer_type": "bool",
        "metric": "Accuracy",
        "example_answer": "yes"
    })

    questions_and_answers.append({
        "question": "Which week for the patient's blood glucose is more stable?",
        "answer": f"week {1 if week1_stats['cv'] < week2_stats['cv'] else 2}",
        "answer_generation_rule": "Calculate the coefficient of variation (CV = SD/Mean * 100) for each week's glucose values and select the one with the lower CV.",
        "answer_instruction": "Return 'week 1' or 'week 2' based on which week has lower glucose CV.",
        "answer_type": "string",
        "metric": "Accuracy",
        "example_answer": "week 1"
    })

    questions_and_answers.append({
        "question": "How is the time in range compared with last week?",
        "answer": "increased" if week2_stats['tir_percent'] > week1_stats['tir_percent'] else ("decreased" if week2_stats['tir_percent'] < week1_stats['tir_percent'] else "no change"),
        "answer_generation_rule": "Calculate % of readings between 70–180 mg/dL for each week and compare.",
        "answer_instruction": "Calculate % of readings between 70–180 mg/dL for each week and compare. Return one of the following: 'increased', 'decreased', or 'no change'.",
        "answer_type": "string",
        "metric": "Accuracy",
        "example_answer": "increased"
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
                    qa["question_id"] = f"pm_14days_{i}"

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
    day = 14
    base_path = f"./SimulationData/morning_runner_openloop_{day}day"
    output_path = "./QA_pairs"
    os.makedirs(output_path, exist_ok=True)

    patient_id = "morning_runner_1"
    input_file = os.path.join(base_path, f"{patient_id}_simulation_data.jsonl")
    output_file = os.path.join(output_path, f"{patient_id}_questions_answers_{day}day.jsonl")
    main(input_file, output_file, include_patient_data=True)