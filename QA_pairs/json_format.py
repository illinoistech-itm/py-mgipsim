import json

def print_structure(data, indent=0):
    prefix = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key} ({type(value).__name__})")
            print_structure(value, indent + 2)
    elif isinstance(data, list):
        print(f"{prefix}[List of {len(data)} items]")
        if data:
            print_structure(data[0], indent + 2)  # Print structure of first item
    else:
        print(f"{prefix}{type(data).__name__}")

# Load and inspect your JSON
with open("morning_runner_1_questions_answers_openloop_insulin.jsonl", "r") as f:
    json_data = json.load(f)
    print_structure(json_data)
