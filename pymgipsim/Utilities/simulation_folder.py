import os, shutil, traceback
from .metadata import SimulationMetaData
from pymgipsim.Utilities.Scenario import load_scenario, save_scenario
from pymgipsim.Utilities.paths import default_settings_path, scenarios_path
from dataclasses import asdict
import pandas
import tqdm


def save_to_xls(state_results, state_names, state_units, destination, no_progress_bar, faults_label):
    for subject_index in tqdm.tqdm(range(state_results.shape[0]), disable=no_progress_bar):
        df = pandas.DataFrame(state_results[subject_index].T,
                              columns=[f"{i} ({j})" for i, j in zip(state_names, state_units)])

        df['faults_label'] = faults_label

        df.to_excel(destination, sheet_name=f'Patient_{subject_index}', engine='xlsxwriter')

def get_most_recent_folder_in_directory(directory: str) -> str:
    """
    Get the most recent folder in a given directory based on modification time.

    Parameters:
    - directory: str
        Directory path.

    Returns:
    - str
        Path of the most recent folder.
    """
    all_subdirs = [os.path.join(directory, d) for d in os.listdir(directory)]
    most_recent_dir = max(all_subdirs, key=os.path.getmtime)
    
    return most_recent_dir


def create_simulation_name(start_time_stamp: str) -> str:
    """
    Create a simulation name based on the start timestamp.

    Parameters:
    - start_time_stamp: str
        Start timestamp of the simulation.

    Returns:
    - str
        Generated simulation name.
    """
    return f"Simulation {start_time_stamp}"

def create_simulation_folder(results_folder_directory: str, simulation_name: str) -> str:
    """
    Create a simulation folder and subfolders for figures.

    Parameters:
    - results_folder_directory: str
        Path to the results folder.
    - simulation_name: str
        Name of the simulation.

    Returns:
    - str
        Path to the created simulation folder.
    """
    folder_path = os.path.join(results_folder_directory, simulation_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    elif os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path) 

    os.mkdir(os.path.join(folder_path, "figures"))

    return folder_path

def create_simulation_results_folder(results_directory: str) -> tuple[str, str, str, str]:
    """
    Create a simulation results folder with metadata.

    Parameters:
    - results_directory: str
        Path to the results directory.

    Returns:
    - tuple
        Tuple containing start timestamp, Python system information, simulation name, and folder path.
    """
    
    start_time_stamp = SimulationMetaData.generate_timestamp()
    python_system_information = SimulationMetaData.generate_system_information()

    simulation_name = create_simulation_name(start_time_stamp)

    folder_path = create_simulation_folder(results_directory, simulation_name)

    return start_time_stamp, python_system_information, simulation_name, folder_path


def load_settings_file(args, results_folder_path):

    if args.scenario_name:

        print(args.scenario_name)

        absolute_path_to_scenario = os.path.abspath(os.path.join(scenarios_path, args.scenario_name))

        if '.json' not in absolute_path_to_scenario[-5:]:
            absolute_path_to_scenario += '.json'

        if os.path.exists(absolute_path_to_scenario):

            try:
                settings_file = load_scenario(absolute_path_to_scenario)
                print(f"..... Loading scenario {absolute_path_to_scenario}")

            except Exception as error:
                print(f"..... Could not open file {absolute_path_to_scenario}")
                print(traceback.format_exc())

    else:
        settings_file = load_scenario(os.path.join(default_settings_path, "scenario_default.json"))

    save_scenario(os.path.join(results_folder_path, "simulation_settings.json"), asdict(settings_file))

    assert settings_file is not None

    return settings_file