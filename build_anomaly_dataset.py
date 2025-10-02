"""
Builds the anomaly-detection QA dataset in one pass.

Pipeline:
1) Load and preprocess simulation inputs per patient (carbs/exercise/insulin + BG).
2) Import ground-truth question functions from `generate_ground_truth_rule` and
   parse their docstrings into QA metadata.
3) Execute each question function on each patient's dataframe to produce answers.
4) Write:
   - QA json (per-question, per-patient) for inspection
   - Final JSONL records combining input_context + qa_pairs (per patient)

Usage:
    python build_anomaly_dataset.py \
        --base_dir SimulationResults \
        --out_inputs_dir SimulationData \
        --qa_json QA_VanillaMPC.json \
        --out_jsonl VanillaMPC_anomaly_detection.jsonl \
        --num_patients 20
"""

import argparse
import inspect
import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# === GT rules module ===
import generate_ground_truth_rule as ggt

# ----------------------------- Config & Logging ------------------------------

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ----------------------------- Utilities ------------------------------------

def extract_ad_number(fname: str) -> int:
    """Return the numeric part after '_ad' for ordering; large fallback if none."""
    return int(fname.split("_ad")[-1]) if "_ad" in fname else 9999


def parse_docstring(doc: Optional[str], fallback_id: str) -> Tuple[str, str, str, str, str, str]:
    """
    Parse a function docstring that follows the pattern:

        ad_XX: <question text>
        Answer_generation_rule: ...
        Metric: ...
        Answer_type: ...
        Answer_instruction: ...

    Returns:
        (qid, qtext, rule, metric, answer_type, answer_instruction)
    """
    if not doc:
        return fallback_id, "", "", "", "", ""

    qid = fallback_id
    qtext = ""
    rule = ""
    metric = ""
    answer_type = ""
    answer_instruction = ""

    for raw in doc.strip().splitlines():
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("ad_") and ":" in line:
            parts = line.split(":", 1)
            qid = parts[0].strip()
            qtext = parts[1].strip()
        elif lower.startswith("answer_generation_rule:"):
            rule = line.split(":", 1)[1].strip()
        elif lower.startswith("metric:"):
            metric = line.split(":", 1)[1].strip()
        elif lower.startswith("answer_type:"):
            answer_type = line.split(":", 1)[1].strip()
        elif lower.startswith("answer_instruction:"):
            answer_instruction = line.split(":", 1)[1].strip()

    return qid, qtext, rule, metric, answer_type, answer_instruction


def generate_example_answer(answer_type: str) -> Any:
    """
    Produce a synthetic example answer consistent with the declared answer_type.
    Keep it simple; this is for documentation checks only.
    """
    t = (answer_type or "").lower()
    if "float" in t:
        return round(random.uniform(0, 100), 1)
    if 'list of {"start"' in t:
        start = random.randrange(0, 30000, 5)
        end = start + random.randrange(5, 55, 5)
        return [{"start": start, "end": end}]
    if "list of int (e.g., days)" in t:
        return random.sample(range(1, 31), 3)
    if "list of int" in t:
        return random.sample(range(0, 30000, 5), 3)
    if t.strip() == "int":
        return random.randint(0, 20)
    if "yes" in t or "no" in t:
        return random.choice(["yes", "no"])
    return None


def format_time_info(mins: float) -> Tuple[int, str]:
    """
    Convert absolute minute index into (day_index, HH:MM).
    Day index starts at 1 and increases monotonically (no weekly reset).
    """
    mins = float(mins)
    day = int(mins // 1440) + 1
    time_of_day = mins % 1440
    hours = int(time_of_day // 60)
    minutes = int(time_of_day % 60)
    return day, f"{hours:02d}:{minutes:02d}"


# ------------------------- Input Extraction (Step 1) -------------------------

def extract_carb_events(payload: Dict[str, Any], person_idx: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    if "meal_carb" in payload["inputs"]:
        mags = payload["inputs"]["meal_carb"]["magnitude"][person_idx]
        times = payload["inputs"]["meal_carb"]["start_time"][person_idx]
        meal_types = ["breakfast", "lunch", "dinner"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "carbs": carbs, "meal_type": meal_types[idx % 3]
            })

    if "snack_carb" in payload["inputs"]:
        mags = payload["inputs"]["snack_carb"]["magnitude"][person_idx]
        times = payload["inputs"]["snack_carb"]["start_time"][person_idx]
        snack_types = ["morning_snack", "afternoon_snack"]
        for idx, (carbs, t) in enumerate(zip(mags, times)):
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "carbs": carbs, "meal_type": snack_types[idx % 2]
            })

    events.sort(key=lambda x: x["time"])
    return events


def extract_exercise_events(payload: Dict[str, Any], person_idx: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for source, label in [("running_speed", "running"), ("cycling_power", "cycling")]:
        if source not in payload["inputs"]:
            continue
        magnitudes = payload["inputs"][source]["magnitude"][person_idx]
        start_times = payload["inputs"][source]["start_time"][person_idx]
        durations = payload["inputs"][source]["duration"][person_idx]
        for mag, t, d in zip(magnitudes, start_times, durations):
            if mag == 0:
                continue
            day, time_str = format_time_info(t)
            events.append({
                "time": t, "day": day, "time_str": time_str,
                "duration": d, "magnitude": mag, "exercise_type": label
            })
    events.sort(key=lambda x: x["time"])
    return events


def extract_insulin_from_csv(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    return {"magnitude": df.iloc[:, 0].tolist()}


def load_bg_df(xlsx_path: str, sheet_name: str = "Patient_0") -> pd.DataFrame:
    """
    Load BG sheet and convert IG (mmol/L) to BG mg/dL = IG * 18
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if "IG (mmol/L)" not in df.columns:
        raise ValueError(f"Missing 'IG (mmol/L)' in {xlsx_path}:{sheet_name}")
    df = df.copy()
    df["BG"] = df["IG (mmol/L)"] * 18
    return df


def build_input_context_for_patient(base_dir: str, patient_id: str) -> Dict[str, Any]:
    """
    Build `input_context` for one patient:
        - carb_events
        - exercise_events (if any)
        - insulin_events (if csv present)
        - bg_mgdl
    """
    patient_dir = os.path.join(base_dir, patient_id)
    simulation_path = os.path.join(patient_dir, "simulation_settings.json")
    bg_path = os.path.join(patient_dir, "model_state_results.xlsx")
    insulin_csv_path = os.path.join(patient_dir, "insulin_input.csv")

    if not os.path.exists(simulation_path) or not os.path.exists(bg_path):
        raise FileNotFoundError(f"{patient_id}: missing simulation_settings.json or model_state_results.xlsx")

    with open(simulation_path, "r") as f:
        payload = json.load(f)

    input_context: Dict[str, Any] = {}
    input_context["carb_events"] = extract_carb_events(payload, 0)

    exercise_events = extract_exercise_events(payload, 0)
    if exercise_events:
        input_context["exercise_events"] = exercise_events

    if os.path.exists(insulin_csv_path):
        input_context["insulin_events"] = extract_insulin_from_csv(insulin_csv_path)
    else:
        logging.warning("%s: insulin_input.csv not found; leaving empty list.", patient_id)
        input_context["insulin_events"] = []

    try:
        bg_df = load_bg_df(bg_path, sheet_name="Patient_0")
        input_context["bg_mgdl"] = bg_df["BG"].tolist()
    except Exception as e:
        logging.error("%s: failed to load BG sheet (%s)", patient_id, e)
        input_context["bg_mgdl"] = []

    return input_context


# ------------------------- QA Generation (Step 2) ----------------------------

def collect_question_funcs() -> Dict[str, Any]:
    funcs = {
        name: func for name, func in inspect.getmembers(ggt, inspect.isfunction)
        if "_ad" in name
    }
    return dict(sorted(funcs.items(), key=lambda x: extract_ad_number(x[0])))


def collect_question_metadata(funcs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for fname, func in funcs.items():
        doc = inspect.getdoc(func)
        qid, qtext, rule, metric, answer_type, answer_instruction = parse_docstring(doc, fallback_id=fname)
        meta[qid] = {
            "function_name": fname,
            "question_id": qid,
            "question_text": qtext,
            "answer_generation_rule": rule,
            "metric": metric,
            "answer_type": answer_type,
            "answer_instruction": answer_instruction,
        }
    return meta


def sanitize_intervals_if_needed(answer: Any) -> Any:
    """
    If answer is a list of intervals [{start, end}], ensure end > start.
    """
    if isinstance(answer, list) and answer:
        for a in answer:
            if isinstance(a, dict) and "start" in a and "end" in a and a["start"] == a["end"]:
                a["end"] = a["end"] + 1
    return answer


def run_qa_for_patient(base_dir: str, patient_id: str, funcs: Dict[str, Any], meta_by_qid: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute all question functions on one patient's dataframe and build qa_pairs.
    Returns:
        (qa_pairs, input_context)
    """
    # Load BG dataframe and allow ggt.preprocess_df to do its work
    xlsx = os.path.join(base_dir, patient_id, "model_state_results.xlsx")
    if not os.path.exists(xlsx):
        logging.warning("Skipped %s: %s not found.", patient_id, xlsx)
        return [], {}

    df = pd.read_excel(xlsx)
    df = ggt.preprocess_df(df)  

    qa_pairs: List[Dict[str, Any]] = []

    for fname, func in funcs.items():
        try:
            ans = func(df)
        except Exception as e:
            logging.error("Error in %s for %s: %s", fname, patient_id, e)
            ans = None

        ans = sanitize_intervals_if_needed(ans)
        ad_num = fname.split("_ad")[-1]
        lookup_id = f"ad_{ad_num}"
        meta = meta_by_qid.get(lookup_id, {
            "function_name": fname,
            "question_id": lookup_id,
            "question_text": ""
        })

        qa_pairs.append({
            "patient_id": patient_id,
            "function_name": fname,
            "question_id": meta["question_id"],
            "question_text": meta["question_text"],
            "answer_generation_rule": meta.get("answer_generation_rule", ""),
            "answer_instruction": meta.get("answer_instruction", ""),
            "answer_type": meta.get("answer_type", ""),
            "metric": meta.get("metric", ""),
            "answer": ans,
            "example_answer": generate_example_answer(meta.get("answer_type", "")),
        })

    # Build input_context after QA; order doesn’t matter but we reuse BG once
    input_context = build_input_context_for_patient(base_dir, patient_id)
    return qa_pairs, input_context


# ------------------------- Packing (Step 3) ----------------------------------

def write_qa_json(all_qa: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(all_qa, f, indent=2)
    logging.info("Wrote QA json: %s", path)


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logging.info("Wrote JSONL: %s", path)


# ------------------------------ CLI -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build anomaly-detection QA dataset.")
    parser.add_argument("--base_dir", type=str, default="SimulationResults", help="Root folder containing Patient_* subfolders.")
    parser.add_argument("--out_inputs_dir", type=str, default="SimulationData", help="Folder to optionally store per-patient input snapshots (JSONL).")
    parser.add_argument("--qa_json", type=str, default="QA_VanillaMPC.json", help="Path to write the QA json (flat list).")
    parser.add_argument("--out_jsonl", type=str, default="VanillaMPC_anomaly_detection.jsonl", help="Final per-patient JSONL with input_context + qa_pairs.")
    parser.add_argument("--num_patients", type=int, default=20, help="How many patients to iterate from 1..N.")
    parser.add_argument("--dump_inputs", action="store_true", help="If set, also dump per-patient input_context JSONL under out_inputs_dir.")

    args = parser.parse_args()
    os.makedirs(args.out_inputs_dir, exist_ok=True)

    funcs = collect_question_funcs()
    meta_by_qid = collect_question_metadata(funcs)

    all_qa_flat: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for i in range(1, args.num_patients + 1):
        pid = f"Patient_{i}"
        logging.info("Processing %s ...", pid)

        qa_pairs, input_context = run_qa_for_patient(args.base_dir, pid, funcs, meta_by_qid)
        if not qa_pairs:
            continue

        # extend flat QA list (mirrors your original QA json)
        all_qa_flat.extend(qa_pairs)

        # combined record
        record = {
            "patient_id": pid,
            "input_context": input_context,
            "qa_pairs": qa_pairs
        }
        records.append(record)

        # optional: dump a per-patient input snapshot (mirrors your SimulationData/*.jsonl)
        if args.dump_inputs:
            out_file = os.path.join(args.out_inputs_dir, f"{pid}_simulation_data.jsonl")
            with open(out_file, "w") as f:
                f.write(json.dumps({"patient_id": pid, **input_context}) + "\n")
            logging.info("Saved input snapshot: %s", out_file)

    # write artifacts
    write_qa_json(all_qa_flat, args.qa_json)
    write_jsonl(records, args.out_jsonl)


if __name__ == "__main__":
    main()
