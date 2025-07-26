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
    carb_events = patient_data["carb_events"]
    insulin_events = patient_data["insulin_events"]
    
    # Blood glucose related calculations per day
    daily_bg = {}
    samples_per_day = 24 * 12  # 288 samples per day (every 5 minutes)
    num_days = len(bg_values) // samples_per_day

    morning_sds = []
    afternoon_sds = []

    for day in range(1, num_days + 1):
        start_idx = (day - 1) * samples_per_day
        end_idx = min(day * samples_per_day, len(bg_values))
        day_bg_values = bg_values[start_idx:end_idx]
        morning_values = day_bg_values[72:144]     # 6:00–12:00
        afternoon_values = day_bg_values[144:216]   # 12:00–18:00

        if len(day_bg_values) == 0:
            continue
            
        # Calculate daily statistics
        day_mean = np.mean(day_bg_values)
        day_max = np.max(day_bg_values)
        day_min = np.min(day_bg_values)
        day_std = np.std(day_bg_values)
        day_cv = (day_std / day_mean) * 100 if day_mean != 0 else 0  # Coefficient of variation

        morning_sd = np.std(morning_values)
        afternoon_sd = np.std(afternoon_values)
        morning_sds.append(morning_sd)
        afternoon_sds.append(afternoon_sd)
        
        # Peak time within the day (in minutes from start of day)
        peak_idx_in_day = np.argmax(day_bg_values)
        peak_time_minutes = peak_idx_in_day * 5
        peak_hours = peak_time_minutes // 60
        peak_mins = peak_time_minutes % 60
        peak_time_str = f"{peak_hours:02d}:{peak_mins:02d}"
        
        # Time in range for this day
        day_in_range = [70 <= bg <= 180 for bg in day_bg_values]
        day_tir_minutes = sum(day_in_range) * 5
        day_tir_hours = day_tir_minutes / 60
        day_tir_percentage = (sum(day_in_range) / len(day_bg_values)) * 100
        
        # Hypoglycemic and hyperglycemic events for this day
        day_hypo_events = 0
        day_hyper_events = 0
        in_hypo = False
        in_hyper = False
        
        for bg in day_bg_values:
            if bg < 70 and not in_hypo:
                in_hypo = True
                day_hypo_events += 1
            elif bg >= 70 and in_hypo:
                in_hypo = False
                
            if bg > 180 and not in_hyper:
                in_hyper = True
                day_hyper_events += 1
            elif bg <= 180 and in_hyper:
                in_hyper = False
        
        # Rapid fluctuations for this day
        glucose_rates = [day_bg_values[i] - day_bg_values[i - 1] for i in range(1, len(day_bg_values))]
        rapid_fluctuations = [abs(rate) > 2 for rate in glucose_rates]
        has_rapid_fluctuations = any(rapid_fluctuations)

        # Insulin events for this day
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
            "mean": round(day_mean, 1),
            "max": round(day_max, 1),
            "min": round(day_min, 1),
            "std": round(day_std, 1),
            "cv": round(day_cv, 1),
            "morning_sd": morning_sd,
            "afternoon_sd": afternoon_sd,
            "peak_time": peak_time_str,
            "peak_value": round(day_max, 1),
            "time_in_range_minutes": day_tir_minutes,
            "time_in_range_hours": round(day_tir_hours, 1),
            "time_in_range_percentage": round(day_tir_percentage, 1),
            "hypo_events": day_hypo_events,
            "hyper_events": day_hyper_events,
            "has_rapid_fluctuations": has_rapid_fluctuations,
            "total_insulin": round(total_insulin, 2),
            "largest_bolus_amount": largest_bolus_amount,
            "largest_bolus_time": largest_bolus_time,
        }

    # meal-related calculations per day
    for meal in carb_events:
        day = int(meal['day'])
        meal_type = meal['meal_type']
        meal_time = meal["time"]
        meal_index = int(meal_time // 5)  # Convert from minutes to index
        carbs = meal['carbs']

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
            peak_time_str = format_time_info(peak_time)

            # Time to return to baseline after peak
            time_to_baseline = None
            if peak_index < len(post_meal_bg) - 1:
                for i in range(peak_index + 1, len(post_meal_bg)):
                    if post_meal_bg[i] <= baseline * 1.1:  # Within 10% above baseline
                        time_to_baseline = (i - peak_index) * 5  # convert to minutes
                        break

            # Initial rise rate (first 30 min)
            initial_window_steps = min(30 // 5, len(post_meal_bg) - 1)  # 6 steps
            initial_rise = post_meal_bg[initial_window_steps] - baseline
            rise_rate = initial_rise / 30  # mg/dL per minute

            day_key = f"day{day}"
            if day_key not in daily_bg:
                daily_bg[day_key] = {}

            if "meal_responses" not in daily_bg[day_key]:
                daily_bg[day_key]["meal_responses"] = {}

            daily_bg[day_key]["meal_responses"][meal_type] = {
                "carbs": carbs,
                "baseline": baseline,
                "peak": peak,
                "peak_time": peak_time_str,
                "time_to_baseline_min": time_to_baseline,
                "rise_rate_per_min": round(rise_rate, 2),
                "spike": round((peak - baseline),1),
            }

            # Track max spike for the day
            spike = peak - baseline
            curr_max = daily_bg[day_key].get("max_spike", -1)
            if spike > curr_max:
                daily_bg[day_key]["max_spike"] = spike
                daily_bg[day_key]["max_spike_meal"] = meal_type

            daily_bg[day_key]["max_spike_response"] = daily_bg[day_key]["meal_responses"][meal_type]

    # activities-related calculations per day
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
        post_sd = np.std(post_bg)
        post_cv = round((post_sd / post_mean) * 100, 2)
        post_nadir_idx = np.argmin(post_bg)
        time_to_lowest_post_exercise = post_nadir_idx * 5  # assuming 5-minute sampling


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
            "post_cv": post_cv,
            "max_drop": baseline-post_nadir,
            "rate_of_change_post": rate_of_change_post,
            "start_time": start_time_str,
            "duration_min": duration,
            "time_to_baseline_min": time_to_baseline,
            "time_to_lowest_post_exercise": time_to_lowest_post_exercise,
            "day": day,
            "exercise_type": exercise_type,
        }


    # stable days
    stable_days = []
    most_variable_day = None
    most_stable_day = None
    max_cv = -1
    min_cv = float('inf')

    for day_key, day_data in daily_bg.items():
        day_num = int(day_key.replace("day", ""))
        cv = day_data["cv"]
        if cv < 36:
            stable_days.append(day_num)

        if cv > max_cv:
            max_cv = cv
            most_variable_day = day_num

        if cv < min_cv:
            min_cv = cv
            most_stable_day = day_num

    # morning/afternoon comparision   
    avg_morning_sd = np.mean(morning_sds)
    avg_afternoon_sd = np.mean(afternoon_sds)

    if avg_morning_sd > avg_afternoon_sd:
        higher_period = "mornings"
    elif avg_afternoon_sd > avg_morning_sd:
        higher_period = "afternoons"
    else:
        higher_period = "equal in both"

    # weekdays/weekends comparision 
    weekday_cvs = [daily_bg[f"day{i}"]["cv"] for i in range(1, 6)]
    avg_cv_weekdays = np.mean(weekday_cvs)
    weekend_cvs = [daily_bg[f"day{i}"]["cv"] for i in range(6, 8)]
    avg_cv_weekends = np.mean(weekend_cvs)

    if avg_cv_weekdays > avg_cv_weekends:
        higher_weekdays_weekends = "weekdays"
    elif avg_cv_weekdays < avg_cv_weekends:
        higher_weekdays_weekends = "weekends"
    else:
        higher_weekdays_weekends = "equal in both"


    # Generate questions and answers
    questions_and_answers = []
    
    # Basic Statistics questions
    questions_and_answers.append({
        "question": "What was the average glucose reading between 2-4pm on Tuesday?",
        "answer": f"{np.mean(daily_bg["day2"]["bg"][168:192]):.1f} mg/dL",
        "answer_generation_rule": "Filter readings for Tuesday 2-4 pm and compute the mean.",
        "answer_instruction": "Return the average glucose value for Tuesday between 2-4pm, rounded to one decimal place, followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "128.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What are the days where glucose are consistently stable?",
        "answer": f"{stable_days}",
        "answer_generation_rule": "Return a list of day indices (1-based) where glucose coefficient of variation is lower than 36%.",
        "answer_instruction": "Return a list of day numbers (1-based indexing) where the coefficient of variation is below 36%. Format as a Python list, e.g., [1, 3, 5].",
        "answer_type": "list",
        "metric": "F1",
        "example_answer": "[1, 3, 5, 7]"
    })

    questions_and_answers.append({
        "question": "On which day was the patient's glucose most variable?",
        "answer": f"{most_variable_day}",
        "answer_generation_rule": "Return the day index (1-based) with the highest coefficient of variation on that day.",
        "answer_instruction": "Return the day number (1-based indexing) with the highest glucose variability as an integer.",
        "answer_type": "int",
        "metric": "Accuracy",
        "example_answer": "4"
    })

    questions_and_answers.append({
        "question": "On which day was the patient's glucose most stable?",
        "answer": f"{most_stable_day}",
        "answer_generation_rule": "Return the day index (1-based) with the lowest coefficient of variation on that day.",
        "answer_instruction": "Return the day number (1-based indexing) with the lowest glucose variability as an integer.",
        "answer_type": "int", 
        "metric": "Accuracy",
        "example_answer": "2"
    })

    # Comparison questions
    questions_and_answers.append({
        "question": "Was glucose variability higher in the mornings or afternoons?",
        "answer": f"{higher_period}",
        "answer_generation_rule": "Calculate standard deviations of glucose values for 6am–12pm and 12pm–6pm periods across all days, and compare the averages.",
        "answer_instruction": "Select one of the following options based on which period has higher glucose variability: 'morning', 'afternoon', or 'equal in both'.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "morning"
    })

    questions_and_answers.append({
        "question": "Are my overnight glucose trends more stable on weekdays or weekend?",
        "answer": f"{higher_weekdays_weekends}",
        "answer_generation_rule": "Compare the glucose coefficient of variation during overnight hours (10pm-6am) on weekdays versus weekends.",
        "answer_instruction": "Select one of the following options based on which period has more stable overnight glucose: 'weekdays', 'weekend', or 'equal in both'.",
        "answer_type": "categorical",
        "metric": "Accuracy", 
        "example_answer": "weekdays"
    })


    # Meal-related questions
    heavy_dinner_days = []

    for day in range(1, num_days + 1):
        day_key = f"day{day}"
        meal_responses = daily_bg.get(day_key, {}).get("meal_responses", {})
        dinner = meal_responses.get("dinner", None)
        if dinner and dinner.get("carbs", 0) >= 100:
            spike = dinner.get("spike", 0)
            heavy_dinner_days.append((day, spike))

    questions_and_answers.append({
        "question": "List all days with a carb-heavy dinner and the corresponding glucose spikes.",
        "answer": heavy_dinner_days,
        "answer_generation_rule": "Find days where the dinner meal has 100 or more carbs and report the glucose spike after dinner as (peak - baseline).",
        "answer_instruction": "Return a list of tuples where each tuple contains the day number and the glucose spike value.",
        "answer_type": "list of tuples (int, float)",
        "metric": "F1",
        "example_answer": "[(1, 85.0), (3, 92.5), (7, 105.3)]"
    })

    questions_and_answers.append({
        "question": "What was the peak glucose level after lunch on Wednesday?",
        "answer": f"{daily_bg['day3']['meal_responses']['lunch']['peak']:.1f} mg/dL",
        "answer_generation_rule": "Find maximum glucose value in the post-lunch window (typically 3 hours) on Wednesday.",
        "answer_instruction": "Return the highest glucose value after lunch on Wednesday as a float with one decimal, followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "165.0 mg/dL"
    })

    questions_and_answers.append({
        "question": "How long did it take for glucose levels to return to baseline after dinner on Monday?",
        "answer": f"{daily_bg['day1']['meal_responses']['dinner']['time_to_baseline_min']} minutes",
        "answer_generation_rule": "Find the first time after the peak when glucose returns to within 10% of the pre-meal baseline on Monday after dinner.",
        "answer_instruction": "Return the number of minutes it took for glucose to return to baseline after Monday dinner.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "90"
    })

    questions_and_answers.append({
        "question": "Which meal caused the highest glucose spike on Friday?",
        "answer": f"{daily_bg['day5']['max_spike_response']}",
        "answer_generation_rule": "Compare 'spike' (peak minus baseline) values for breakfast, lunch, and dinner on Friday. Select the meal with the highest spike.",
        "answer_instruction": "Return the meal name with the highest glucose spike on Friday: 'breakfast', 'lunch', or 'dinner'.",
        "answer_type": "str",
        "metric": "Accuracy",
        "example_answer": "lunch"
    })
    
    questions_and_answers.append({
        "question": "What was the glucose rise rate after the morning snack on Saturday?",
        "answer": f"{daily_bg['day6']['meal_responses']['morning_snack']['rise_rate_per_min']:.2f} mg/dL/min",
        "answer_generation_rule": "Calculate the rate of glucose increase from baseline to peak after the morning snack on Saturday.",
        "answer_instruction": "Return the glucose rise rate per minute after the morning snack on Satday, rounded to two decimal places, followed by 'mg/dL/min'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "1.45 mg/dL/min"
    })

    # Activity-related questions

    questions_and_answers.append({
        "question": "What was the peak glucose level during exercise on Sunday?",
        "answer": f"{exercise_responses['day7']['max_during']:.1f} mg/dL",
        "answer_generation_rule": "Find the maximum glucose level during the exercise period on Sunday.",
        "answer_instruction": "Return the highest glucose value during the running session on Sunday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "125.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the lowest glucose level during exercise on Monday?",
        "answer": f"{exercise_responses['day1']['min_during']:.1f} mg/dL",
        "answer_generation_rule": "Find the minimum glucose level during the exercise period on Monday.",
        "answer_instruction": "Return the lowest glucose value during the cycling session on Monday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "78.0 mg/dL"
    })


    questions_and_answers.append({
        "question": "What was the lowest glucose level after exercise on Tuesday?",
        "answer": f"{exercise_responses['day2']['post_nadir']:.1f} mg/dL",
        "answer_generation_rule": "Find the minimum glucose value within 60 minutes after exercise ends on Tuesday.",
        "answer_instruction": "Return the lowest glucose value after exercise on Tuesday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "85.2 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the glucose rate of change after exercise on Wednesday?",
        "answer": f"{exercise_responses['day3']['rate_of_change_post']:.2f} mg/dL/min",
        "answer_generation_rule": "Calculate the rate of glucose change in the first 60 minutes after exercise on Wednesday.",
        "answer_instruction": "Return the glucose rate of change after exercise on Wednesday, rounded to two decimal places followed by 'mg/dL/min'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "0.87 mg/dL/min"
    })

    questions_and_answers.append({
        "question": "How long did it take for glucose levels to return to baseline after exercise on Thursday?",
        "answer": f"{exercise_responses['day4']['time_to_baseline_min']} minutes",
        "answer_generation_rule": "Find the first time after exercise ends when glucose returns to within 10% of the pre-exercise baseline on Thursday.",
        "answer_instruction": "Return the number of minutes it took for glucose to return to baseline after exercise on Thursday.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "75"
    })

    questions_and_answers.append({
        "question": "What was the patient's average glucose level within 1 hour after reported exercise on Friday?",
        "answer": f"{exercise_responses['day5']['post_mean']:.1f} mg/dL",
        "answer_generation_rule": "Compute the average glucose value for the 60 minutes following the end of exercise on Friday.",
        "answer_instruction": "Return the average glucose level after exercise within 1 hour on Friday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "112.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the patient's average glucose level during reported exercise on Satday?",
        "answer": f"{exercise_responses['day6']['mean_during']:.1f} mg/dL",
        "answer_generation_rule": "Compute the average glucose value during exercise on Satday.",
        "answer_instruction": "Return the average glucose level during exercise on Satday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "110.5 mg/dL"
    })

    questions_and_answers.append({
        "question": "What was the maximum glucose drop following activity on Wednesday?",
        "answer": f"{exercise_responses['day3']['max_drop']:.1f} mg/dL",
        "answer_generation_rule": "Subtract the lowest post-exercise glucose value within 1 hour time window from the pre-exercise baseline on Wednesday.",
        "answer_instruction": "Subtract the lowest post-exercise glucose value within 1 hour time window from the pre-exercise baseline on Wednesday, as a float with one decimal followed by 'mg/dL'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "35.0 mg/dL"
    })

    questions_and_answers.append({
        "question": "How long after exercise did the patient's glucose reach its lowest point on Tuesday last week?",
        "answer": f"{exercise_responses['day2']['time_to_lowest_post_exercise']} minutes",
        "answer_generation_rule": "Find the time difference (in minutes) between exercise end and the post-exercise glucose nadir on Tuesday of Week 1.",
        "answer_instruction": "Return the number of minutes it took to reach the lowest glucose level after exercise on Tuesday last week.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": "45"
    })

    stable_days_after_exercise = []
    for day_key, ex in exercise_responses.items():
        day = int(day_key.replace("day",''))
        if ex['post_cv'] < 36:
            stable_days_after_exercise.append(day)


    questions_and_answers.append({
        "question": "On which days of the week does post-exercise glucose tend to be most stable?",
        "answer": f"{stable_days_after_exercise}",
        "answer_generation_rule": "For each day with exercise, compute the coefficient of variation (CV) of glucose levels in the 60-minute post-exercise window. If CV is below 36%, classify the day as stable",
        "answer_instruction": "Return a list of day index (1-based) for which the 1 hour post-exercise glucose CV was below 36%. ",
        "answer_type": "list of int",
        "metric": "F1",
        "example_answer": "[1, 3, 5]"
    })

    
    # insulin-related questions
    questions_and_answers.append({
        "question": "What was the patient's total daily insulin dose on Tuesday?",
        "answer": f"{daily_bg['day2']['total_insulin']:.2f} units",
        "answer_generation_rule": "Sum all basal and bolus insulin amounts recorded throughout Tuesday.",
        "answer_instruction": "Return the total insulin dose on Tuesday, rounded to two decimal places, followed by 'units'.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": "34.00 units"
    })

    questions_and_answers.append({
        "question": "When did the patient receive their largest insulin bolus on Sunday?",
        "answer": f"{daily_bg['day7']['largest_bolus_time']}",  
        "answer_generation_rule": "Find the insulin bolus event with the highest insulin amount on Sunday and return its time.",
        "answer_instruction": "Return the time of the largest bolus on Sunday in the format <HH:MM>.",
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
        "question": "Does the patient use more insulin on weekends in the this week?",
        "answer": f"{"Yes" if avg_weekend_insulin > avg_weekday_insulin else "No"}",
        "answer_generation_rule": (
            "Calculate the average insulin doses for weekend (day 6 and day 7) of the the week. "
            "Compare with the average daily insulin use on weekdays (day 1 to 5) this week. "
            "If the weekend average is greater than weekday average, return 'Yes'; otherwise, return 'No'."
        ),
        "answer_instruction": (
            "Return 'Yes' if the average of insulin use on weekends is higher than the average insulin use on weekdays in the the week; "
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
                    qa["question_id"] = f"pm_7days_{i}"

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
    base_path = './SimulationData/morning_runner_openloop_7day'
    output_path = os.path.join(base_path, "QA")
    os.makedirs(output_path, exist_ok=True)

    patient_id = "morning_runner_1"
    input_file = os.path.join(base_path, f"{patient_id}_simulation_data.jsonl")
    output_file = os.path.join(output_path, f"{patient_id}_questions_answers.jsonl")
    main(input_file, output_file, include_patient_data=True)