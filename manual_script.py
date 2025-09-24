import subprocess
from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder

from pymgipsim.Interface.parser import generate_parser_cli
from pymgipsim.InputGeneration.activity_settings import activity_args_to_scenario

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.generate_plots import generate_plots_main
from pymgipsim.generate_results import generate_results_main

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
    args.controller_name = (
        "OpenAPS"  # Select controller folder in pymgipsim/Controller/...
    )
    args.model_name = "T1DM.ExtHovorka" # Select Hovorka model
    # args.patient_names = ["Patient_1"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.patient_names = [
        "Patient_1",
        "Patient_2",
        "Patient_3",
        "Patient_4",
        "Patient_5",
        "Patient_6",
        "Patient_7",
        "Patient_8",
        "Patient_9",
        "Patient_10",
        "Patient_11",
        "Patient_12",
        "Patient_13",
        "Patient_14",
        "Patient_15",
        "Patient_16",
        "Patient_17",
        "Patient_18",
        "Patient_19",
        "Patient_20",
    ]
    args.running_speed = 0.0 # Turn off physical activity
    args.plot_patient = (
        0  # Plots patient glucose, intakes, heartrate - commented to plot all patients
    )
    args.breakfast_carb_range = [30, 60]
    args.am_snack_carb_range = [10, 20]
    args.lunch_carb_range = [30, 60]
    args.pm_snack_carb_range = [10, 20]
    args.dinner_carb_range = [30, 60]
    args.random_seed = 100
    args.number_of_days = 30
    args.sampling_time = 5

    activity_args_to_scenario(settings_file, args)
    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

    model,_ = generate_results_main(scenario_instance = settings_file, args = vars(args), results_folder_path = results_folder_path)

    # Generate plots for all patients
    all_figures = []
    for i in range(len(args.patient_names)):
        args.plot_patient = i
        print(f"Generating plots for {args.patient_names[i]} (index {i})")
        figures = generate_plots_main(results_folder_path, args)
        all_figures.append(figures)
