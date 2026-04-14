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
    
    ################## Overall Statistics over 30 Days ####################
    bg_values = patient_data["bg_mgdl"]
    bg_time = [i * 5 for i in range(len(bg_values))] # The data were sampled every 5 mins

    # Basic statistics
    avg_glucose = np.mean(bg_values)
    max_glucose = np.max(bg_values)
    min_glucose = np.min(bg_values)
    
    max_glucose_time = format_time_info(np.argmax(bg_values) * 5) # Example: (day, time) (1, '01:40')
    max_glucose_time_minutes = np.argmax(bg_values) * 5  # Convert to timestamp in minutes
    std_glucose = np.std(bg_values)
    
    # Target range calculations
    in_range = [70 <= bg <= 180 for bg in bg_values]
    time_in_range_minutes = sum(in_range) * 5
    time_in_range_hours = time_in_range_minutes / 60
    time_in_range_percentage = (sum(in_range) / len(bg_values)) * 100  

    above_range = [bg > 180 for bg in bg_values]
    print(sum(above_range))
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

    glucose_rates = [(bg_values[i] - bg_values[i-1]) / 5 for i in range(1, len(bg_values))]
    rapid_fluctuations = [abs(rate) > 2 for rate in glucose_rates]
    has_rapid_fluctuations = any(rapid_fluctuations)


    ################ Daily Statitics ###################
    daily_bg = {}
    samples_per_day = 24 * 12  # 288 samples per day (every 5 minutes)
    num_days = len(bg_values) // samples_per_day

    morning_sds = []
    afternoon_sds = []
    evening_sds = []

    for day in range(1, num_days + 1):
        start_idx = (day - 1) * samples_per_day
        end_idx = min(day * samples_per_day, len(bg_values))
        day_bg_values = bg_values[start_idx:end_idx]
        morning_values = day_bg_values[72:144]     # 6:00–12:00
        afternoon_values = day_bg_values[144:216]   # 12:00–18:00
        evening_values = day_bg_values[216:288]  # 18:00-24:00

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
        evening_sd = np.std(evening_values)
        morning_sds.append(morning_sd)
        afternoon_sds.append(afternoon_sd)
        evening_sds.append(evening_sd)
        
        # Peak time within the day (in minutes from start of day)
        peak_idx_in_day = np.argmax(day_bg_values)
        peak_idx_global = start_idx + peak_idx_in_day
        peak_time_minutes = peak_idx_global * 5

        # Convert to hours and minutes within a 24h day
        peak_hours = (peak_time_minutes // 60) % 24
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
            "peak_time_minutes": peak_time_minutes,  # Add timestamp version
            "peak_value": round(day_max, 1),
            "time_in_range_minutes": day_tir_minutes,
            "time_in_range_hours": round(day_tir_hours, 1),
            "time_in_range_percentage": round(day_tir_percentage, 1),
            "hypo_events": day_hypo_events,
            "hyper_events": day_hyper_events,
            "has_rapid_fluctuations": has_rapid_fluctuations,
        }

    
    # meal-related calculations per day
    carb_events = patient_data["carb_events"]
    meal_responses = {}
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

        post_meal_bg = bg_values[meal_index +1 : meal_index + post_meal_window + 1]

        if len(post_meal_bg) > 1:
            peak_index = np.argmax(post_meal_bg)
            peak = post_meal_bg[peak_index]
            peak_time = int((meal_time // 5 + peak_index) * 5) # convert from steps to minutes
            peak_time_str = format_time_info(peak_time)

            # Time to return to baseline after peak
            time_to_baseline = None
            for i in range(peak_index + 1, len(post_meal_bg)):
                if post_meal_bg[i] <= baseline * 1.1:  # within 10% of baseline
                    time_to_baseline = (i - peak_index) * 5
                    break


            # Initial rise rate (first 30 min)
            initial_window_steps = min(30 // 5, len(post_meal_bg) - 1)  # 6 steps
            initial_rise = post_meal_bg[initial_window_steps] - baseline
            rise_rate = initial_rise / 30  # mg/dL per minute

            day_key = f"day{day}"
            if day_key not in meal_responses:
                meal_responses[day_key] = {}

            if meal_type not in meal_responses[day_key]:
                meal_responses[day_key][meal_type] = {
                    "carbs": carbs,
                    "baseline": baseline,
                    "peak": peak,
                    "peak_time": peak_time_str,
                    "peak_time_minutes": peak_time,  # Add timestamp version
                    "time_to_baseline_min": time_to_baseline,
                    "rise_rate_per_min": round(rise_rate, 1),
                    "spike": round((peak - baseline),1),
                }

            # Track max spike for the day
            spike = peak - baseline
            curr_max = meal_responses[day_key].get("max_spike", -1)
            if spike > curr_max:
                meal_responses[day_key]["max_spike"] = spike
                meal_responses[day_key]["max_spike_meal"] = meal_type

            # meal_responses[day_key]["max_spike_response"] = meal_responses[day_key][meal_type]

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


        # Baseline before exercise
        baseline = bg_values[start_index]

        # During exercise
        during_bg = bg_values[start_index : end_index]
        min_during = np.min(during_bg)
        max_during = np.max(during_bg) 
        mean_during = np.mean(during_bg)

        # Post-exercise window 2-hour
        post_window_len = min(120 // 5, len(bg_values) - end_index - 1)
        post_bg = bg_values[end_index + 1: end_index + post_window_len + 1]
        post_peak = np.max(post_bg)
        post_nadir = np.min(post_bg)
        post_mean = np.mean(post_bg)
        post_sd = np.std(post_bg)
        post_cv = round((post_sd / post_mean) * 100, 2)
        post_nadir_idx = np.argmin(post_bg)
        time_to_lowest_post_exercise = post_nadir_idx * 5  # assuming 5-minute sampling


        # Rate of change (over exercise duration)
        if len(during_bg) > 1:
            rate_of_change = round(((during_bg[-1] - baseline) / duration),1)
        else:
            rate_of_change = 0


        # Time to return to baseline post-exercise compared to pre-exercise baseline
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
            "rate_of_change": rate_of_change,
            "post_peak": post_peak,
            "post_nadir": post_nadir,
            "post_mean": post_mean,
            "post_cv": post_cv,
            "max_drop": baseline-post_nadir,
            "start_time": start_time_str,
            "start_time_minutes": start_time,  # Add timestamp version
            "duration_min": duration,
            "time_to_baseline_min": time_to_baseline,
            "time_to_lowest_post_exercise": time_to_lowest_post_exercise,
            "day": day,
            "exercise_type": exercise_type,
        }

    
    # Generate questions and answers
    questions_and_answers = []
    
    ########## Overall Statistics #############
    # Basic Statistics questions
    questions_and_answers.append({
        "question_text": "What was the patient's average blood glucose level?",
        "answer": round(avg_glucose, 1),
        "answer_generation_rule": "Calculate the mean of all blood glucose values.",
        "answer_instruction": "Return the average of all blood glucose values, rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 80.5,
    })
   
    questions_and_answers.append({
        "question_text": "What was the highest blood glucose level recorded?",
        "answer": round(max_glucose, 1),
        "answer_generation_rule": "Find the maximum value in the blood glucose array.",
        "answer_instruction": "Scan all glucose readings to identify the maximum value. Report it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE", 
        "example_answer": 130.0
    })

    questions_and_answers.append({
        "question_text": "What was the lowest blood glucose level recorded?",
        "answer": round(min_glucose, 1),
        "answer_generation_rule": "Find the minimum value in the blood glucose array.",
        "answer_instruction": "Scan all glucose readings to identify the minimum value. Report it as a float rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 50.0
    })

    questions_and_answers.append({
        "question_text": "When did the patient experience their highest blood glucose peak?",
        "answer": max_glucose_time_minutes,
        "answer_generation_rule": "Find the index of the maximum glucose value and convert to time in minutes.",
        "answer_instruction": "Scan all glucose readings to identify the maximum value. Locate the index of its first occurrence and multiply by 5 to convert to minutes, and report the corresponding minutes.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 860
    })

    questions_and_answers.append({
        "question_text": "How many hours did the patient spend in the target range (70-180 mg/dL)?",
        "answer": round(time_in_range_hours, 1),
        "answer_generation_rule": "Count minutes where glucose values are between 70-180 mg/dL, then convert to hours.",
        "answer_instruction": "Count the number of glucose readings between 70 and 180 mg/dL (inclusive), multiply by 5 to get minutes, then convert to hours and round to one decimal.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 5.5
    })

    questions_and_answers.append({
        "question_text": "How many hours did the patient spend above the target range (> 180 mg/dL)?",
        "answer": round(time_above_range_hours, 1),
        "answer_generation_rule": "Count minutes where glucose values are above 180 mg/dL (do not include 180), then convert to hours.",
        "answer_instruction": "Identify all glucose readings strictly greater than 180 mg/dL (exclude 180), count the number of such readings, multiply by 5 to convert the count to minutes, then divide by 60 to convert to hours and round to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 5.5
    })

    questions_and_answers.append({
        "question_text": "How many hours did the patient spend below the target range (< 70 mg/dL)?",
        "answer": round(time_below_range_hours, 1),
        "answer_generation_rule": "Count minutes where glucose values are below 70 mg/dL (not including 70), then convert to hours.",
        "answer_instruction": "Identify all glucose readings strictly less than 70 mg/dL (exclude 70), count the number of such readings, multiply by 5 to convert the count to minutes, then divide by 60 to convert to hours and round to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 5.5
    })

    questions_and_answers.append({
        "question_text": "How many hypoglycemic events (BG < 70 mg/dL) did the patient experience?",
        "answer": hypo_events,
        "answer_generation_rule": "Count the total number of blood glucose readings that are below 70 mg/dL (not including 70). Return the count as an integer.",
        "answer_instruction": "Identify all glucose readings strictly less than 70 mg/dL (exclude 70), count the total number of such readings, and return this count as an integer.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 3
    })

    questions_and_answers.append({
        "question_text": "How many hyperglycemic events (BG > 180 mg/dL) did the patient experience?",
        "answer": sum(above_range),
        "answer_generation_rule": "Count the total number of blood glucose readings that are above 180 mg/dL (not including 180). Return the count as an integer.",
        "answer_instruction": "Identify all glucose readings strictly greater than 180 mg/dL (exclude 180), count the total number of such readings, and return this count as an integer.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 3
    })

    questions_and_answers.append({
        "question_text": "What was the standard deviation of the patient's glucose levels?",
        "answer": round(std_glucose, 1),
        "answer_generation_rule": "Calculate the standard deviation of all blood glucose values.",
        "answer_instruction": "Return the standard deviation of glucose values, rounded to one decimal place",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 2.0
    })

    questions_and_answers.append({
        "question_text": "What's the time-in-range percentage for this patient?",
        "answer": round(time_in_range_percentage, 1),
        "answer_generation_rule": "Divide time in range (70-180 mg/dL) by total time, multiply by 100.",
        "answer_instruction": "Calculate the proportion of glucose readings between 70 and 180 mg/dL (inclusive), multiply by 100 to get percentage, and round to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 36.0
    })

    questions_and_answers.append({
        "question_text": "What's the time-above-range percentage for this patient?",
        "answer": round(time_above_range_percentage, 1),
        "answer_generation_rule": "Divide time above range (> 180 mg/dL not including 180) by total time, multiply by 100.",
        "answer_instruction": "Calculate the proportion of glucose readings strictly greater than 180 mg/dL (exclude 180) relative to the total number of readings, multiply by 100 to obtain the percentage, and round to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 38.0
    })

    questions_and_answers.append({
        "question_text": "What's the time-below-range percentage for this patient?",
        "answer": round(time_below_range_percentage, 1),
        "answer_generation_rule": "Divide time below range (< 70 mg/dL not including 70) by total time, multiply by 100.",
        "answer_instruction": "Calculate the proportion of glucose readings strictly less than 70 mg/dL (exclude 70) relative to the total number of readings, multiply by 100 to obtain the percentage, and round to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 40.0
    })

    questions_and_answers.append({
        "question_text": "Did the patient experience any rapid glucose fluctuations (>2 mg/dL/min)?",
        "answer": "Yes" if has_rapid_fluctuations else "No",
        "answer_generation_rule": "Calculate rate of change between consecutive readings, check if any exceed 2 mg/dL/min.",
        "answer_instruction": "For each pair of consecutive glucose readings, compute the rate of change by subtracting the earlier value from the later value and dividing by the time interval in minutes between the readings, take the absolute value of each rate, check whether any rate exceeds 2 mg/dL/min, and return 'Yes' if at least one does, otherwise return 'No'.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "Yes"
    })


    ########## Daily Statistics #############
    # Basic Statistics questions
    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the average glucose reading between 2-4pm on {day_name}?",
        "answer": round(np.mean(daily_bg[day_key]["bg"][168:192]), 1),
        "answer_generation_rule": f"Filter readings for 2-4 pm on {day_name} and compute the mean.",
        "answer_instruction": f"Return the average glucose value for {day_name} between 2-4pm, rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 128.5
    })

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
        higher_period = "morning"
    elif avg_afternoon_sd > avg_morning_sd:
        higher_period = "afternoon"
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
        higher_weekdays_weekends = "weekend"
    else:
        higher_weekdays_weekends = "equal in both"

    questions_and_answers.append({
        "question_text": "What are the days where glucose are consistently stable?",
        "answer": stable_days,
        "answer_generation_rule": "Return a list of day indices (1-based) where glucose coefficient of variation is lower than 36.",
        "answer_instruction": "For each day, calculate the coefficient of variation by dividing the standard deviation of that day’s glucose values by the mean glucose for that day and multiplying by 100, identify all days where this value is less than 36, and return their day indices using 1-based numbering as a list, e.g., [1, 3, 5].",
        "answer_type": "list",
        "metric": "F1",
        "example_answer": [1, 3, 15, 27]
    })

    questions_and_answers.append({
        "question_text": "On which day was the patient's glucose most variable?",
        "answer": most_variable_day,
        "answer_generation_rule": "Return the day index (1-based) with the highest coefficient of variation on that day.",
        "answer_instruction": "For each day, calculate the coefficient of variation by dividing the standard deviation of that day’s glucose values by the mean glucose for that day and multiplying by 100, compare these daily values across all days, and return the 1-based day index of the day with the highest coefficient of variation.",
        "answer_type": "int",
        "metric": "Accuracy",
        "example_answer": 4
    })

    questions_and_answers.append({
        "question_text": "On which day was the patient's glucose most stable?",
        "answer": most_stable_day,
        "answer_generation_rule": "Return the day index (1-based) with the lowest coefficient of variation on that day.",
        "answer_instruction": "For each day, calculate the coefficient of variation by dividing the standard deviation of that day’s glucose values by the mean glucose for that day and multiplying by 100, compare these daily values across all days, and return the 1-based day index of the day with the lowest coefficient of variation.",
        "answer_type": "int", 
        "metric": "Accuracy",
        "example_answer": 2
    })

    # Comparison questions
    questions_and_answers.append({
        "question_text": "Was glucose variability higher in the mornings or afternoons?",
        "answer": higher_period,
        "answer_generation_rule": "Calculate standard deviations of glucose values for 6am–12pm and 12pm–6pm periods across all days, and compare the averages.",
        "answer_instruction": "For each day, separate glucose readings into the 6am–12pm period and the 12pm–6pm period, calculate the standard deviation of glucose values for each period on each day, compute the average standard deviation across all days for each time period, and compare the two averages to determine which period shows greater glucose variability. Select one of the following options based on which period has higher glucose variability: 'morning', 'afternoon', or 'equal in both'.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "morning"
    })

    questions_and_answers.append({
        "question_text": "Are my glucose trends more stable on weekdays or weekend for the first week?",
        "answer": higher_weekdays_weekends,
        "answer_generation_rule": "Compare the glucose coefficient of variation on weekdays versus weekends.",
        "answer_instruction": "Select one of the following options based on which period has more stable glucose: 'weekdays', 'weekend', or 'equal in both'.",
        "answer_type": "categorical",
        "metric": "Accuracy", 
        "example_answer": "weekdays"
    })


    # Meal-related questions
    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"How did the patient's blood glucose respond to breakfast on {day_name}?",
        "answer": {
            'Baseline': round(meal_responses[day_key]['breakfast']['baseline'],1),
            'Peak': round(meal_responses[day_key]['breakfast']['peak'],1), 
            'Peak_time': meal_responses[day_key]['breakfast']['peak_time_minutes']
        },
        "answer_generation_rule": "Compare baseline glucose at meal time to maximum value in post-meal window (3 hours) on {day_name}.",
        "answer_instruction": f"Return a dictionary containing the baseline glucose value (the glucose value immediately before the meal), the peak glucose value, and the time of the peak (in minutes from the start) within the 3-hour post-meal window on {day_name}. Use the keys 'Baseline', 'Peak', and 'Peak_time'.",
        "answer_type": "dict",
        "metric": "{'Baseline': 'MAE', 'Peak': 'MAE', 'Peak_time': 'MAE'}",
        "example_answer": {'Baseline': 95.0, 'Peak': 160.0, 'Peak_time': 525},
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the peak glucose level after lunch on {day_name}?",
        "answer": round(meal_responses[day_key]['lunch']['peak'], 1),
        "answer_generation_rule": f"Find maximum glucose value in the post-lunch window (3 hours) on {day_name}.",
        "answer_instruction": f"Identify the data corresponding to {day_name}, locate the lunch time for that day, select all glucose readings within the 3-hour window following lunch, find the maximum glucose value among those readings, and return it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 165.0
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"How long did it take for glucose levels to return to baseline after dinner on {day_name}?",
        "answer": meal_responses[day_key]['dinner']['time_to_baseline_min'],
        "answer_generation_rule": f"Find first time after peak when glucose returns to within 10% of pre-meal baseline on {day_name}, and calculate how long it took in minutes.",
        "answer_instruction": f"Identify {day_name} and locate the dinner time, record the baseline glucose as the value immediately before dinner, select all glucose readings within the 3-hour post-dinner window, find the peak glucose value and its timestamp within this window, then examine subsequent readings after the peak to find the first time point where glucose falls within ±10% of the baseline value, compute the time difference in minutes between the meal time (or peak time if specified) and this return time, and return this duration; if no such point exists within the 3-hour window, return None.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 90
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"
    questions_and_answers.append({
        "question_text": f"Which meal caused the highest glucose spike on {day_name}?",
        "answer": meal_responses[day_key]['max_spike_meal'],
        "answer_generation_rule": f"Compare spike (peak minus baseline) values for all meals in post-meal window (3 hours) to find largest increase on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the times of breakfast, lunch, and dinner, for each meal record the baseline glucose as the value immediately before the meal, select glucose readings within the 3-hour post-meal window, compute the spike as the difference between the maximum glucose value in the window and the baseline, compare the spikes across the three meals, and return the meal name ('breakfast', 'lunch', or 'dinner') corresponding to the largest spike.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "lunch"
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the glucose rise rate after the morning snack on {day_name}?",
        "answer": meal_responses[day_key]['morning_snack']['rise_rate_per_min'],
        "answer_generation_rule": f"Calculate rate of increase from baseline to 30-minute post-meal glucose on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the morning snack time, record the baseline glucose as the value immediately before the snack, find the glucose value at 30 minutes after the snack, compute the rate of increase as (glucose at 30 minutes minus baseline) divided by 30 minutes, and return the result rounded to one decimal places.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 1.45
    })

    heavy_dinner_days = []

    for day in range(1, num_days + 1):
        day_key = f"day{day}"
        dinner = meal_responses.get(day_key, {}).get("dinner", None)
        if dinner and dinner.get("carbs", 0) >= 80:
            spike = dinner.get("spike", 0)
            heavy_dinner_days.append((day, spike))

    questions_and_answers.append({
        "question_text": "List all days with a carb-heavy dinner and the corresponding glucose spikes.",
        "answer": heavy_dinner_days,
        "answer_generation_rule": "Find days where the dinner meal has 80 or more carbs and report the glucose spike after dinner as (peak - baseline) in a 3 hour time window.",
        "answer_instruction": "Find days where the dinner meal has 80 or more carbs. For each such day, calculate the glucose spike within the 3-hour post-dinner window, defined as the difference between the peak glucose level and the baseline value (the glucose measurement immediately before dinner). Return the results as a list of tuples, where each tuple contains the day number (1–30) and the spike value rounded to one decimal place.",
        "answer_type": "list of tuples (int, float)",
        "metric": "F1",
        "example_answer": [(1, 85.0), (12, 92.5), (21, 105.3)]
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the peak glucose level after lunch on {day_name}?",
        "answer": round(meal_responses[day_key]['lunch']['peak'], 1),
        "answer_generation_rule": f"Find maximum glucose value in the post-lunch window (3 hours) on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the lunch time, select all glucose readings within the 3-hour post-lunch window, find the maximum glucose value among those readings, and return it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 165.0
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"Which meal caused the highest glucose spike on {day_name}?",
        "answer": meal_responses[day_key]['max_spike_meal'],
        "answer_generation_rule": f"Compare 'spike' (peak - baseline) values for breakfast, lunch, and dinner in post-meal window (3 hours) on {day_name}. Select the meal with the highest spike.",
        "answer_instruction": f"Identify {day_name} and locate breakfast, lunch, and dinner times, for each meal record the baseline glucose as the value immediately before the meal, select glucose readings within the 3-hour post-meal window, compute the spike as the difference between the maximum glucose value in the window and the baseline, compare the spikes across the three meals, and return the meal name ('breakfast', 'lunch', or 'dinner') corresponding to the largest spike.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "lunch"
    })
    

    # Activity-related questions

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the peak glucose level during exercise on {day_name}?",
        "answer": round(exercise_responses[day_key]['max_during'], 1),
        "answer_generation_rule": f"Find the maximum glucose level during the exercise period on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the start and end times of the exercise session, select all glucose readings that occur during this exercise period, find the maximum glucose value among those readings, and return it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 125.5
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the lowest glucose level during exercise on {day_name}?",
        "answer": round(exercise_responses[day_key]['min_during'], 1),
        "answer_generation_rule": f"Find the minimum glucose level during the exercise period on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the start and end times of the exercise session, select all glucose readings that occur during this exercise period, find the minimum glucose value among those readings, and return it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 78.0
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the lowest glucose level after exercise on {day_name}?",
        "answer": round(exercise_responses[day_key]['post_nadir'], 1),
        "answer_generation_rule": f"Find the minimum glucose value within 180 minutes after exercise ends on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the end time of the exercise session, select all glucose readings within the 180-minute window following the end of exercise, find the minimum glucose value among those readings, and return it as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 85.2
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the glucose rate of change after exercise on {day_name}?",
        "answer": exercise_responses[day_key]['rate_of_change'],
        "answer_generation_rule": f"Calculate the rate of glucose change in the first 60 minutes after exercise on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the exercise session, record the baseline glucose as the value immediately before exercise starts, find the first glucose measurement at or after the exercise end time, compute the rate of change by subtracting the baseline value from this post-exercise glucose value and dividing by the elapsed time in minutes between the two measurements, and return the result rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 0.87
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"How long did it take for glucose levels to return to baseline after exercise on {day_name}?",
        "answer": exercise_responses[day_key]['time_to_baseline_min'],
        "answer_generation_rule": f"Find the first time after exercise ends when glucose returns to within 10% of the pre-exercise baseline on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the exercise session, record the pre-exercise baseline as the glucose value immediately before exercise starts, select all glucose readings within the 2-hour window after exercise ends, find the first reading in this window whose glucose value falls within ±10% of the baseline, calculate the time difference in minutes between the exercise end time and that reading, and return this duration; if no such reading occurs within the 2-hour window, return None.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 75
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the patient's average glucose level within 1 hour after reported exercise on {day_name}?",
        "answer": round(exercise_responses[day_key]['post_mean'], 1),
        "answer_generation_rule": f"Compute the average glucose value for the 60 minutes following the end of exercise on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the end time of the exercise session, select all glucose readings within the 1-hour window immediately following exercise end, calculate the average of those glucose values, and return the result as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 112.5
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the patient's average glucose level during reported exercise on {day_name}?",
        "answer": round(exercise_responses[day_key]['mean_during'], 1),
        "answer_generation_rule": f"Compute the average glucose value during exercise on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the start and end times of the exercise session, select all glucose readings that occur during the exercise period, calculate the average of those glucose values, and return the result as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 110.5
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"What was the maximum glucose drop following activity on {day_name}?",
        "answer": round(exercise_responses[day_key]['max_drop'], 1),
        "answer_generation_rule": f"Subtract the lowest post-exercise glucose value within 1 hour time window from the pre-exercise baseline on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the exercise session, record the pre-exercise baseline as the glucose value immediately before exercise starts, select all glucose readings within the 2-hour window after exercise ends, find the minimum glucose value in that window, subtract this minimum value from the baseline value, and return the result as a float rounded to one decimal place.",
        "answer_type": "float",
        "metric": "MAE",
        "example_answer": 35.0
    })

    random_day = random.randint(1, 30)
    day_key = f"day{random_day}"
    
    week = (random_day - 1) // 7 + 1
    day_in_week = (random_day - 1) % 7 + 1
    day_name = f"week {week}, day {day_in_week}"

    questions_and_answers.append({
        "question_text": f"How long after exercise did the patient's glucose reach its lowest point on {day_name}?",
        "answer": exercise_responses[day_key]['time_to_lowest_post_exercise'],
        "answer_generation_rule": f"Find the time difference (in minutes) between exercise end and the post-exercise glucose nadir on {day_name}.",
        "answer_instruction": f"Identify {day_name} and locate the end time of the exercise session, select all glucose readings within the 2-hour window after exercise ends, find the minimum glucose value in that window and the first time it occurs, calculate the time difference in minutes between the exercise end time and that reading, and return the duration in minutes.",
        "answer_type": "int",
        "metric": "MAE",
        "example_answer": 45
    })

    stable_days_after_exercise = []
    for day_key, ex in exercise_responses.items():
        day = int(day_key.replace("day",''))
        if ex['post_cv'] < 36:
            stable_days_after_exercise.append(day)


    questions_and_answers.append({
        "question_text": "On which days does post-exercise glucose tend to be stable?",
        "answer": stable_days_after_exercise,
        "answer_generation_rule": "For each day with exercise, compute the coefficient of variation (CV) of glucose levels in the 60-minute post-exercise window. If CV is below 36, classify the day as stable",
        "answer_instruction": "Iterate over all days that contain an exercise session, for each day locate the end time of exercise, select all glucose readings within the 2-hour post-exercise window, calculate the coefficient of variation by dividing the standard deviation of those glucose values by their mean and multiplying by 100, identify the days where this value is less than 36, and return their 1-based day indices as a list.",
        "answer_type": "list of int",
        "metric": "F1",
        "example_answer": [1, 3, 15, 28]
    })

    # week level questions
    # Calculate weekly metrics (assuming 7-day weeks, ignoring the last 2 days))
    bg_values = patient_data["bg_mgdl"]  # list of bg values every 5 mins
    readings_per_day = 288
    readings_per_week = readings_per_day * 7

    week1_bg = bg_values[:readings_per_week]
    week2_bg = bg_values[readings_per_week:readings_per_week * 2]
    week3_bg = bg_values[readings_per_week * 2:readings_per_week * 3]
    week4_bg = bg_values[readings_per_week * 3:readings_per_week * 4]

    def get_metrics(week_bg):
        week_array = np.array(week_bg)
        mean = np.mean(week_array)
        std = np.std(week_array)
        cv = round((std / mean) * 100, 1) if mean > 0 else None

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
    week3_stats = get_metrics(week3_bg)
    week4_stats = get_metrics(week4_bg)

    first_week, second_week = random.sample(range(1, 5), 2)

    questions_and_answers.append({
        "question_text": f"Did the patient spend less time in hypoglycemia in week {first_week} than in week {second_week}?",
        "answer": "yes" if eval(f"week{first_week}_stats['hypoglycemia_minutes']") < eval(f"week{second_week}_stats['hypoglycemia_minutes']") else "no",
        "answer_generation_rule": f"Sum minutes with glucose < 70 mg/dL in week {first_week} and week {second_week}, then compare.",
        "answer_instruction": f"Select all glucose readings from week {first_week} and count how many are strictly less than 70 mg/dL, convert that count to minutes of hypoglycemia, then do the same for week {second_week}, compare the total hypoglycemia minutes between the two weeks, and return 'yes' if week {first_week} has fewer hypoglycemia minutes than week {second_week}; otherwise return 'no'.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "yes"
    })

    first_week, second_week = random.sample(range(1, 5), 2)

    questions_and_answers.append({
        "question_text": f"Which week had more stable blood glucose: week {first_week} or week {second_week}?",
        "answer": f"week {first_week}" if eval(f"week{first_week}_stats['cv']") < eval(f"week{second_week}_stats['cv']") else f"week {second_week}",
        "answer_generation_rule": f"Compare CV (SD/mean) of glucose readings from week {first_week} and week {second_week}.",
        "answer_instruction": f"Collect all glucose readings from week {first_week} and calculate their coefficient of variation by dividing the standard deviation by the mean and multiplying by 100, repeat the same calculation for week {second_week}, compare the two CV values, and return {first_week} if week {first_week} has the lower coefficient of variation; otherwise return {second_week}.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": f"week {first_week}"
    })

    first_week, second_week = random.sample(range(1, 5), 2)

    questions_and_answers.append({
        "question_text": f"How did the patient's time in range change from week {first_week} to week {second_week}?",
        "answer": (
            "increased" if eval(f"week{second_week}_stats['tir_percent']") > eval(f"week{first_week}_stats['tir_percent']")
            else "decreased" if eval(f"week{second_week}_stats['tir_percent']") < eval(f"week{first_week}_stats['tir_percent']")
            else "no change"
        ),
        "answer_generation_rule": f"Compare % of readings in 70–180 mg/dL range between week {first_week} and week {second_week}. Return 'increased', 'decreased', or 'no change' depending on how time in range changed.",
        "answer_instruction": "Collect all glucose readings from week {first_week} and calculate the percentage of readings that fall within the 70–180 mg/dL range. Repeat the same calculation for week {second_week}. Compare the two percentages and return 'increased' if the percentage for week {second_week} is higher, 'decreased' if it's lower, or 'no change' if they are equal.",
        "answer_type": "categorical",
        "metric": "Accuracy",
        "example_answer": "increased"
    })

    return questions_and_answers

def convert_np(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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
                    keys_to_include = ['carb_events', 'insulin_events', 'exercise_events', 'bg_mgdl']
                    results["input_context"] = {k: patient_data.get(k) for k in keys_to_include}

                
                f_out.write(json.dumps(results, default=convert_np) + '\n')
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
    num_patients = 20
    scenario_name = "cycling"
    base_path = f"./SimulationData/{scenario_name}"
    output_path = f"./QA_pairs/{scenario_name}"
    os.makedirs(output_path, exist_ok=True)

    for num in range(num_patients):
        patient_id = f"Patient_{num}"
        input_file = os.path.join(base_path, f"{patient_id}_simulation_data.jsonl")
        output_file = os.path.join(output_path, f"{patient_id}_questions_answers.jsonl")
        main(input_file, output_file, include_patient_data=True)