import subprocess
import pandas as pd
from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder

from pymgipsim.Interface.parser import generate_parser_cli
from pymgipsim.InputGeneration.activity_settings import activity_args_to_scenario

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.generate_plots import generate_plots_main
from pymgipsim.generate_results import generate_results_main

from pymgipsim.faultsGeneration import generate_faults

if __name__ == '__main__':
    """ Parse Arguments  """
    args = generate_parser_cli().parse_args()

    """ Initialization """
    subprocess.run(['python', 'initialization.py'])

    """ Define Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

    """ Load Scenario """
    settings_file = simulation_folder.load_settings_file(args, results_folder_path)

    # Programatically define scenario
    args.controller_name = "HCL0" # Select controller folder in pymgipsim/Controller/...
    args.model_name = "T1DM.ExtHovorka" # Select Hovorka model
    args.patient_names = ["Patient_3", "Patient_4"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    # physical activity
    args.running_speed      = [5.0]
    args.running_start_time = [382.0]
    args.running_duration   = [30.0]
    args.running_incline    = [0.5]
    args.cycling_start      = [596.0]
    args.cycling_duration   = [20.0]
    args.cycling_power      = [90.0]

    args.plot_patient = 0 # Plots patient glucose, intakes, heartrate
    args.breakfast_carb_range = [80, 120]
    args.am_snack_carb_range = [10, 20]
    args.lunch_carb_range = [80, 120]
    args.pm_snack_carb_range = [10, 20]
    args.dinner_carb_range = [80, 120]
    args.random_seed = 100

    args.number_of_days = 30

    args.to_excel = True

    activity_args_to_scenario(settings_file, args)

    # faults injection args
    faults_spec = 'pymgipsim/faultsGeneration/faults_specification.csv'
    simulation_start_time = pd.Timestamp('2023-01-01 00:00:00')

    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

        faults_input = generate_faults.generate_faults(faults_file=faults_spec, simulation_days=args.number_of_days, simulation_start_time=simulation_start_time)

    model,faults_label = generate_results_main(scenario_instance = settings_file, args = vars(args), results_folder_path = results_folder_path, faults_array=faults_input)

    figures = generate_plots_main(results_folder_path, args, faults_label)