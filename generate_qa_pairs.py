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
    
    # Extract relevant data
    # patient_id = patient_data["patient_id"]
    bg_values = patient_data["bg_mgdl"]
    bg_time = [i * 5 for i in range(len(bg_values))] # The data were sampled every 5 mins

    meals = patient_data["carb_events"]
    insulin_events = patient_data["insulin_events"]

    # Basic statistics
    avg_glucose = np.mean(bg_values)
    max_glucose = np.max(bg_values)
    min_glucose = np.min(bg_values)
    
    max_glucose_time = format_time_info(np.argmax(bg_values) * 5) # Example: day, time 1, '01:40'
    std_glucose = np.std(bg_values)
    
    # Target range calculations
    in_range = [70 <= bg <= 180 for bg in bg_values]
    time_in_range_minutes = sum(in_range) * 5
    time_in_range_hours = time_in_range_minutes / 60
    time_in_range_percentage = (sum(in_range) / len(bg_values)) * 100  

    above_range = [bg > 180 for bg in bg_values]
    time_above_range_minutes = sum(above_range) * 5
    time_above_range_hours = time_above_range_minutes / 60
    time_above_range_percentage = (sum(above_range) / len(bg_values)) * 100

    below_range = [bg < 70 for bg in bg_values]
    time_below_range_minutes = sum(below_range) * 5
    time_below_range_hours = time_below_range_minutes / 60
    time_below_range_percentage = (sum(below_range) / len(bg_values)) * 100
    
    # Hypoglycemic/hyperglycemic events
    hypo_events = 0
    hyper_events = 0
    in_hypo = False
    in_hyper = False
    
    for bg in bg_values:
        if bg < 70 and not in_hypo:
            in_hypo = True
            hypo_events += 1
        elif bg >= 70 and in_hypo:
            in_hypo = False
            
        if bg > 180 and not in_hyper:
            in_hyper = True
            hyper_events += 1
        elif bg <= 180 and in_hyper:
            in_hyper = False
    
    # Glucose fluctuations
    glucose_rates = [bg_values[i] - bg_values[i-1] for i in range(1, len(bg_values))]
    rapid_fluctuations = [abs(rate) > 2 for rate in glucose_rates]
    has_rapid_fluctuations = any(rapid_fluctuations)

    # day level statistics
    bg_values = patient_data["bg_mgdl"]
    bg_time = [i * 5 for i in range(len(bg_values))]

    
    # Meal-related calculations
    meal_responses = {}
    carb_events = patient_data["carb_events"]

    for meal in carb_events:
        day = int(meal['day'])
        meal_type = f"{meal['meal_type']}"
        meal_time = meal["time"]
        meal_index = int(meal_time // 5)  # Convert from minutes to index
        
        # Skip if meal time is outside data range
        if meal_index >= len(bg_values):
            continue

        baseline = bg_values[meal_index]
        post_meal_window = min(180 // 5, len(bg_values) - meal_index - 1)  # 36 time steps = 3 hours

        if post_meal_window <= 0:
            continue

        post_meal_bg = bg_values[meal_index : meal_index + post_meal_window + 1]

        if len(post_meal_bg) > 1:
            peak_index = np.argmax(post_meal_bg)
            peak = post_meal_bg[peak_index]
            peak_time = meal_time + peak_index * 5  # convert from steps to minutes
            peak_day, peak_time_str = format_time_info(peak_time)

            # Time to return to baseline after peak
            time_to_baseline = None
            if peak_index < len(post_meal_bg) - 1:
                for i in range(peak_index + 1, len(post_meal_bg)):
                    if post_meal_bg[i] <= baseline * 1.1:  # Within 10% above baseline
                        time_to_baseline = (i - peak_index) * 5  # convert to minutes
                        break

            # Initial rise rate (first 30 min)
            initial_window_steps = min(30 // 5, len(post_meal_bg) - 1) # 6 steps
            initial_rise = post_meal_bg[initial_window_steps] - baseline
            rise_rate = initial_rise / 30  # mg/dL per minute

            meal_responses[meal_type] = {
                "baseline": baseline,
                "peak": peak,
                "peak_time": peak_time_str, # HH:MM
                "time_to_baseline_min": time_to_baseline,
                "rise_rate_per_min": rise_rate
            }

    # Find meal with highest spike
    max_spike_meal = None
    max_spike = 0

    for meal_type, response in meal_responses.items():
        spike = response["peak"] - response["baseline"]
        if spike > max_spike:
            max_spike = spike
            max_spike_meal = meal_type

    # Exercise-related calculations
    exercise_responses = {}
    exercise_events = patient_data["exercise_events"]

    for ex in exercise_events:
        day = int(ex["day"])
        exercise_type = ex["exercise_type"]
        start_time = ex["time"]
        duration = ex["duration"]
        end_time = start_time + duration
        start_index = int(start_time // 5)
        end_index = int(end_time // 5)

        # Skip if event is out of range
        if start_index >= len(bg_values) or end_index >= len(bg_values):
            continue

        # Baseline before exercise
        baseline = bg_values[start_index]

        # During exercise
        during_bg = bg_values[start_index : end_index + 1]
        min_during = np.min(during_bg)
        max_during = np.max(during_bg) 
        mean_during = np.mean(during_bg)

        # Post-exercise window 60 mins
        post_window_len = min(60 // 5, len(bg_values) - end_index - 1)
        post_bg = bg_values[end_index : end_index + post_window_len + 1]
        post_peak = np.max(post_bg)
        post_nadir = np.min(post_bg)
        post_mean = np.mean(post_bg)

        # Rate of change (over exercise duration)
        if len(during_bg) > 1:
            rate_of_change_during = (during_bg[-1] - baseline) / duration
        else:
            rate_of_change_duing = 0

        # Rate of change after exercise
        initial_post_change = post_bg[post_window_len] - bg_values[end_index]
        rate_of_change_post = initial_post_change / 60 

        # Time to return to baseline post-exercise
        time_to_baseline = None
        for i, val in enumerate(post_bg):
            if abs(val - baseline) <= baseline * 0.1:  # within 10%
                time_to_baseline = i * 5  # minutes
                break

        # Time formatting
        start_day, start_time_str = format_time_info(start_time)
        day_key = f"day{day}"
        exercise_responses[day_key] = {
            "baseline": baseline,
            "min_during": min_during,
            "max_during": max_during,
            "mean_during": mean_during,
            "rate_of_change_during": rate_of_change_during,
            "post_peak": post_peak,
            "post_nadir": post_nadir,
            "post_mean": post_mean,
            "max_drop": baseline-post_nadir,
            "rate_of_change_post": rate_of_change_post,
            "start_time": start_time_str,
            "duration_min": duration,
            "time_to_baseline_min": time_to_baseline,
            "day": day,
            "exercise_type": exercise_type,
        }


    # Insulin-related calculations. Only valid if we use OpenLoop. 
    total_insulin = sum(event["dosage"] for event in insulin_events)
    
    largest_bolus = max((event for event in insulin_events if event["insulin_type"] == "bolus_insulin"), key=lambda x: x["dosage"])
    largest_bolus_time = f"largest_bolus['time_str']" 
    largest_bolus_amount = largest_bolus["dosage"]
    
    basal_events = [event for event in insulin_events if event["insulin_type"] == "basal_insulin"]
    
    # Generate questions and answers
    questions_and_answers = []
    
    # Basic Statistics questions
    questions_and_answers.append({
        "question_text": "What was the patient's average blood glucose level?",
        "answer": f"{avg_glucose:.1f} mg/dL",
        "answer_generation_rule": "Calculate the mean of all blood glucose values.",
        "answer_instruction": "Return the average of all blood glucose values, rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "80.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the highest blood glucose level recorded?",
        "answer": f"{max_glucose:.1f} mg/dL",
        "answer_generation_rule": "Find the maximum value in the blood glucose array.",
        "answer_instruction": "Identify the highest blood glucose value and report it as a float rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "130.0 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the lowest blood glucose level recorded?",
        "answer": f"{min_glucose:.1f} mg/dL",
        "answer_generation_rule": "Find the minimum value in the blood glucose array.",
        "answer_instruction": "Identify the lowest blood glucose value and report it as a float rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "50.0 mg/dL"
    })

    questions_and_answers.append({
        "question": "When did the patient experience their highest blood glucose peak?",
        "answer": f"{max_glucose_time}",
        "answer_generation_rule": "Find the index of the maximum glucose value and convert to time format.",
        "answer_instruction": "Return the time of the highest glucose reading in the format HH:MM.",
        "answer_type": "time_str",
        "metric": "Accuracy",
        "example_answer": "14:20"
    })

    questions_and_answers.append({
        "question": "How many hours did the patient spend in the target range (70-180 mg/dL)?",
        "answer": f"{time_in_range_hours:.1f} hours",
        "answer_generation_rule": "Count minutes where glucose values are between 70-180 mg/dL, then convert to hours.",
        "answer_instruction": "Report the total hours spent with glucose in the 70–180 mg/dL range, rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "5.5 hours"
    })

    questions_and_answers.append({
        "question": "How many hypoglycemic events (BG < 70 mg/dL) did the patient experience?",
        "answer": f"{hypo_events}",
        "answer_generation_rule": "Count transitions from normal to hypoglycemic state (BG < 70 mg/dL).",
        "answer_instruction": "Return an integer count of distinct episodes where glucose dropped below 70 mg/dL.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "3"
    })

    questions_and_answers.append({
        "question": "How many hyperglycemic events (BG > 180 mg/dL) did the patient experience?",
        "answer": f"{hyper_events}",
        "answer_generation_rule": "Count transitions from normal to hyperglycemic state (BG > 180 mg/dL).",
        "answer_instruction": "Return an integer count of distinct episodes where glucose rose above 180 mg/dL.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "3"
    })

    questions_and_answers.append({
        "question": "What was the standard deviation of the patient's glucose levels?",
        "answer": f"{std_glucose:.1f} mg/dL",
        "answer_generation_rule": "Calculate the standard deviation of all blood glucose values.",
        "answer_instruction": "Return the standard deviation of glucose values, rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "2.0 mg/dL"
    })

    questions_and_answers.append({
        "question": "What's the time-in-range percentage for this patient?",
        "answer": f"{time_in_range_percentage:.1f}%",
        "answer_generation_rule": "Divide time in range (70-180 mg/dL) by total time, multiply by 100.",
        "answer_instruction": "Return the percentage of time spent in 70–180 mg/dL range",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "36.0%"
    })

    questions_and_answers.append({
        "question": "What's the time-above-range percentage for this patient?",
        "answer": f"{time_above_range_percentage:.1f}%",
        "answer_generation_rule": "Divide time above range (> 180 mg/dL) by total time, multiply by 100.",
        "answer_instruction": "Return the percentage of time above 180 mg/dL",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "38.0%"
    })

    questions_and_answers.append({
        "question": "What's the time-below-range percentage for this patient?",
        "answer": f"{time_below_range_percentage:.1f}%",
        "answer_generation_rule": "Divide time below range (< 70 mg/dL) by total time, multiply by 100.",
        "answer_instruction": "Return the percentage of time below 70 mg/dL, rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "40.0%"
    })

    questions_and_answers.append({
        "question": "Did the patient experience any rapid glucose fluctuations (>2 mg/dL/min)?",
        "answer": "Yes" if has_rapid_fluctuations else "No",
        "answer_generation_rule": "Calculate rate of change between consecutive readings, check if any exceed 2 mg/dL/min.",
        "answer_instruction": "Return 'Yes' if any glucose change exceeds 2 mg/dL/min, otherwise return 'No'.",
        "answer_type": "Yes or No",
        "metric": "Accuracy",
        "example_answer": "Yes"
    })


    # Meal-related questions
    breakfast_meals = [meal for meal in meal_responses.keys() if meal.startswith('breakfast')]
    if breakfast_meals:
        breakfast_meal = breakfast_meals[0]
        questions_and_answers.append({
            "question": "How did the patient's blood glucose respond to breakfast?",
            "answer": f"'Baseline': '{meal_responses[breakfast_meal]['baseline']:.1f} mg/dL', 'Peak': '{meal_responses[breakfast_meal]['peak']:.1f} mg/dL', 'Peak_time': '{meal_responses[breakfast_meal]['peak_time']}'",
            "answer_generation_rule": "Compare baseline glucose at meal time to maximum value in post-meal window.",
            "answer_instruction": "Return the baseline, peak glucose value, and the time of the peak as: 'Baseline: <value> mg/dL, Peak: <value> mg/dL, Peak_time: <HH:MM>'.",
            "answer_type": "dict",
            "metric": "{'Baseline': 'MAE', 'Peak': 'MAE', 'Peak_time': 'Accuracy'}",
            "example_answer": "'Baseline': '95.0 mg/dL', 'Peak': '160.0 mg/dL', 'Peak_time': '08:45'"
        })

    lunch_meals = [meal for meal in meal_responses.keys() if meal.startswith('lunch')]
    if lunch_meals:
        lunch_meal = lunch_meals[0]
        questions_and_answers.append({
            "question": "What was the peak glucose level after lunch?",
            "answer": f"{meal_responses[lunch_meal]['peak']:.1f} mg/dL",
            "answer_generation_rule": "Find maximum glucose value in the post-lunch window (typically 3 hours).",
            "answer_instruction": "Return the highest glucose value after lunch as a float with one decimal, followed by 'mg/dL'.",
            "answer_type": "float",
            "metric": "MAE",
            "example_answer": "165.0 mg/dL"
        })

    dinner_meals = [meal for meal in meal_responses.keys() if meal.startswith('dinner')]
    if dinner_meals and meal_responses[dinner_meals[0]]["time_to_baseline_min"] is not None:
        dinner_meal = dinner_meals[0]
        questions_and_answers.append({
            "question": "How long did it take for glucose levels to return to baseline after dinner?",
            "answer": f"{meal_responses[dinner_meal]['time_to_baseline_min']} minutes",
            "answer_generation_rule": "Find first time after peak when glucose returns to within 10% of pre-meal baseline.",
            "answer_instruction": "Return the number of minutes it took for glucose to return to baseline after dinner.",
            "answer_type": "int",
            "metric": "MAE",
            "example_answer": "90"
        })

    if max_spike_meal:
        questions_and_answers.append({
            "question": "Which meal caused the highest glucose spike?",
            "answer": f"{max_spike_meal}",
            "answer_generation_rule": "Compare peak minus baseline values for all meals to find largest increase.",
            "answer_instruction": "Return the meal name, 'breakfast', 'lunch', or 'dinner'",
            "answer_type": "str",
            "metric": "Accuracy",
            "example_answer": "lunch"
        })

    morning_snack_meals = [meal for meal in meal_responses.keys() if meal.startswith('morning_snack')]
    if morning_snack_meals:
        morning_snack_meal = morning_snack_meals[0]
        questions_and_answers.append({
            "question": "What was the glucose rise rate after the morning snack?",
            "answer": f"{meal_responses[morning_snack_meal]['rise_rate_per_min']:.2f} mg/dL/min",
            "answer_generation_rule": "Calculate rate of increase from baseline to 30-minute post-meal glucose.",
            "answer_instruction": "Return the glucose rise rate per minute after the morning snack, rounded to two decimal places'.",
            "answer_type": "float",
            "metric": "MAE",
            "example_answer": "1.45 mg/dL/min"
        })

    # Exercise-related questions
    questions_and_answers.append({
        "question": "What was the peak glucose level during exercise?",
        "answer": f"{exercise_responses['day1']['max_during']:.1f} mg/dL",
        "answer_generation_rule": "Find the maximum glucose level during the exercise period.",
        "answer_instruction": "Return the highest glucose value during the running session, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "125.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the lowest glucose level during exercise?",
        "answer": f"{exercise_responses['day1']['min_during']:.1f} mg/dL",
        "answer_generation_rule": "Find the minimum glucose level during the exercise period.",
        "answer_instruction": "Return the lowest glucose value during the cycling session, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "78.0 mg/dL"
    })


    questions_and_answers.append({
        "question": "What was the lowest glucose level after exercise?",
        "answer": f"{exercise_responses['day1']['post_nadir']:.1f} mg/dL",
        "answer_generation_rule": "Find the minimum glucose value within 60 minutes after exercise ends.",
        "answer_instruction": "Return the lowest glucose value after cycling, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "85.2 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the glucose rate of change after exercise?",
        "answer": f"{exercise_responses['day1']['rate_of_change_post']:.2f} mg/dL/min",
        "answer_generation_rule": "Calculate the rate of glucose change in the first 60 minutes after exercise.",
        "answer_instruction": "Return the glucose rate of change after walking, rounded to two decimal places followed by 'mg/dL/min'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "0.87 mg/dL/min"
    })

    questions_and_answers.append({
        "question": "How long did it take for glucose levels to return to baseline after exercise?",
        "answer": f"{exercise_responses['day1']['time_to_baseline_min']} minutes",
        "answer_generation_rule": "Find the first time after exercise ends when glucose returns to within 10% of the pre-exercise baseline.",
        "answer_instruction": "Return the number of minutes it took for glucose to return to baseline after cycling.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "75"
    })

    questions_and_answers.append({
        "question": "What was the patient's average glucose level within 1 hour after reported exercise?",
        "answer": f"{exercise_responses['day1']['post_mean']:.1f} mg/dL",
        "answer_generation_rule": "Compute the average glucose value for the 60 minutes following the end of exercise.",
        "answer_instruction": "Return the average glucose level after exercise within 1 hour, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "112.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the patient's average glucose level during reported exercise?",
        "answer": f"{exercise_responses['day1']['mean_during']:.1f} mg/dL",
        "answer_generation_rule": "Compute the average glucose value during exercise.",
        "answer_instruction": "Return the average glucose level during exercise, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "110.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the maximum glucose drop following activity?",
        "answer": f"{exercise_responses['day1']['max_drop']:.1f} mg/dL",
        "answer_generation_rule": "Subtract the lowest post-exercise glucose value within 1 hour time window from the pre-exercise baseline.",
        "answer_instruction": "Subtract the lowest post-exercise glucose value within 1 hour time window from the pre-exercise baseline, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "35.0 mg/dL"
    })


    # Insulin-related questions
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
            "question_text": "How did basal rates change throughout the day?",
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
                    qa["question_id"] = f"pm_{i}"
                
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
    os.makedirs(f'./SimulationData/Morning_runner_1day_OpenLoop/QA/', exist_ok=True)
    input_file="./SimulationData/Morning_runner_1day_OpenLoop/morning_runner_1_simulation_data.jsonl"
    output_file = "./SimulationData/Morning_runner_1day_OpenLoop/QA/morning_runner_1_questions_answers.jsonl"
    main(input_file, output_file, include_patient_data=True)