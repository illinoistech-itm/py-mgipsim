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
    args.controller_name = "HCL0" # Select controller folder in pymgipsim/Controller/...
    args.model_name = "T1DM.ExtHovorka" # Select Hovorka model
    args.patient_names = ["Patient_1", "Patient_3"] # Select Patient in pymgipsim/VirtualPatient/Models/T1DM/ExtHovorka/Patients
    args.plot_patient = 0 # Plots patient glucose, intakes, heartrate
    args.sampling_time = 5

    ''' 
    ###########Default Setting###########
    args.breakfast_carb_range = [80, 120]
    args.am_snack_carb_range = [10, 20]
    args.lunch_carb_range = [80, 120]
    args.pm_snack_carb_range = [10, 20]
    args.dinner_carb_range = [80, 120]

    args.breakfast_time_range = [420, 540]
    args.lunch_time_range = [12, 14]
    args.dinner_time_range = [18, 20]
    args.am_snack_time_range = [10, 11]
    args.pm_snack_time_range = [15, 16]
    args.running_speed = 0.0
    args.cycling_power = 0.0

    ##########Late Heavy Eater##########
    args.am_snack_time_range = [11, 12]
    args.lunch_time_range = [13, 15]
    args.pm_snack_time_range = [16, 17]
    args.dinner_time_range = [19, 21]
    args.breakfast_carb_range = 0.0
    args.am_snack_carb_range = [15, 25]
    args.lunch_carb_range = [100, 140]
    args.pm_snack_carb_range = [15, 25]
    args.dinner_carb_range = [100, 140]
    args.running_speed = 0.0
    args.cycling_power = 0.0

    ############Morning Runner############
    args.running_start_time = [360, 420] # 6-7am
    args.breakfast_time_range = [480, 540] # 8-9am
    args.running_duration = [30, 60]
    args.running_incline = [0.0, 6.0]
    args.running_speed = [1.7, 7.0] 
    '''
 
    #####Lighter Eater with Cycling Hobby#####
    args.breakfast_carb_range = [60, 90] 
    args.lunch_carb_range = [60, 90]
    args.dinner_carb_range = [60, 90]
    args.running_speed = 0.0
    args.cycling_start_time = [960, 1020] # 16-17. snack_pm(15-16) dinner(18-20)
    args.cycling_duration = [20, 60]
    args.cycling_power = [75, 200] 
    
    '''
    ######Erratic Schedule – Skips Lunch######
    args.lunch_carb_range = 0.0
    args.running_speed = 0.0
    args.cycling_power = 0.0
    ''' 

    args.random_seed = 100 
    args.to_excel = True # Store BG values
  
    activity_args_to_scenario(settings_file, args)

    # Or random_scenario = None
    # All target: meal_carb, meal_start_time, snack_carb, snack_start_time
    #             cycling_power, cycling_start_time, cycling_duration
    #             running_speed, running_start_time, running_duration
    # All method: heavy: Increase 10-40% of magnitude or duration
    #             light: Decrease 10-40% of magnitude or duration
    #             early: 1-2 hour before original start_time
    #             delayed: 1-2 hour after original start_time
    #             skipped: Set 0 of magnitude
    # Random seed is same as args.random_seed
    random_scenario = {
        'target': ['meal_carb', 'meal_start_time', 'running_speed', 'running_start_time'],
        'method': ['heavy', 'delayed', 'skipped'],
        'intensity': 0.1
        }

    if not args.scenario_name:

        settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

        settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path, random_scenario=random_scenario)


    model,_ = generate_results_main(scenario_instance = settings_file, args = vars(args), results_folder_path = results_folder_path)

    figures = generate_plots_main(results_folder_path, args)

    # python3 manual_script.py -d 30