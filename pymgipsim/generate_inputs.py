import pymgipsim.VirtualPatient.Models as Models
from pymgipsim.InputGeneration.parsers import generate_input_parser
from pymgipsim.InputGeneration.generate_carb_signal import generate_carb_events
from pymgipsim.InputGeneration.carb_energy_settings import generate_carb_absorption
from pymgipsim.InputGeneration.meal_energy_content import calculate_daily_energy_intake, estimate_g_cho_from_energy_intake
from pymgipsim.InputGeneration.activity_settings import generate_activities
from pymgipsim.InputGeneration.energy_expenditure_settings import generate_energy_expenditure
from pymgipsim.InputGeneration.insulin_settings import generate_basal_insulin, generate_bolus_insulin
from pymgipsim.InputGeneration.heart_rate_settings import generate_heart_rate
from pymgipsim.InputGeneration.generate_multiscale_inputs import generate_multiscale_inputs, generate_bodyweight_events

from pymgipsim.Utilities.paths import default_settings_path, results_path
from pymgipsim.Utilities.Scenario import scenario, inputs, input_generation, load_scenario, save_scenario
from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.Settings.parser import generate_settings_parser
from pymgipsim.VirtualPatient.parser import generate_virtual_subjects_parser
from pymgipsim.Utilities import simulation_folder

from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.InputGeneration.carb_energy_settings import make_carb_settings

import argparse, json, pprint
from dataclasses import asdict
from pymgipsim.Controllers.parser import controller_args_to_scenario
import os
from pymgipsim.Utilities.random_scenarios import randomize_events


def generate_inputs_main(scenario_instance: scenario, args: argparse.Namespace, results_folder_path: str, random_scenario=None) -> argparse.Namespace:

    if not args.no_print:
        print(f">>>>> Generating Input Signals")

    if scenario_instance.settings.simulator_name == "MultiScaleSolver":
        multi_scale_input_generation = input_generation(
            fraction_cho_intake=args.fraction_cho_intake,
            fraction_cho_as_snack=args.fraction_cho_as_snack,
            net_calorie_balance=args.net_calorie_balance
        )
        scenario_instance.input_generation.fraction_cho_intake = multi_scale_input_generation.fraction_cho_intake
        scenario_instance.input_generation.fraction_cho_as_snack = multi_scale_input_generation.fraction_cho_as_snack
        scenario_instance.input_generation.net_calorie_balance = multi_scale_input_generation.net_calorie_balance
        scenario_instance.inputs = inputs()
        """ Update the grams of carbs """
        calculate_daily_energy_intake(scenario_instance, args)
        estimate_g_cho_from_energy_intake(scenario_instance, args)

        """ Calculate the multiscale inputs """

    controller_args_to_scenario(scenario_instance, args)
    make_carb_settings(scenario_instance, args)

    scenario_instance.inputs = inputs()
    scenario_instance.inputs.meal_carb, scenario_instance.inputs.snack_carb = generate_carb_events(scenario_instance, args)
    # Introduce randomness before generate bolus
    if random_scenario:
        meal_target = [t for t in random_scenario['target'] if
                       t in ['meal_carb', 'meal_start_time', 'snack_carb', 'snack_start_time']]
        if any(meal_target):
            scenario_instance.inputs = randomize_events(scenario_instance.inputs, target=meal_target,
                                                        random_way=random_scenario['method'],
                                                        random_state=args.random_seed,
                                                        intensity=random_scenario['intensity'])

    match scenario_instance.patient.model.name:
        case Models.T1DM.IVP.Model.name:
            scenario_instance.inputs.bolus_insulin = generate_bolus_insulin(scenario_instance, args)
            scenario_instance.inputs.basal_insulin = generate_basal_insulin(scenario_instance, args)
            scenario_instance.inputs.taud = generate_carb_absorption(scenario_instance, args)
        
        case Models.T1DM.ExtHovorka.Model.name:
            scenario_instance.inputs.bolus_insulin = generate_bolus_insulin(scenario_instance, args)
            scenario_instance.inputs.basal_insulin = generate_basal_insulin(scenario_instance, args)
            scenario_instance.inputs.running_speed, scenario_instance.inputs.running_incline,\
            scenario_instance.inputs.cycling_power = generate_activities(scenario_instance, args)
            # Introduce randomness before generate heart rate
            if random_scenario:
                exercise_target = [t for t in random_scenario['target'] if t in
                               ['cycling_power', 'cycling_start_time', 'cycling_duration',
                                'running_speed', 'running_start_time', 'running_duration']]
                if any(exercise_target):
                    scenario_instance.inputs = randomize_events(scenario_instance.inputs, target=exercise_target,
                                                                random_way=random_scenario['method'],
                                                                random_state=args.random_seed,
                                                                intensity=random_scenario['intensity'])

            scenario_instance.inputs.heart_rate, scenario_instance.inputs.METACSM = generate_heart_rate(scenario_instance, args)
            scenario_instance.inputs.energy_expenditure = generate_energy_expenditure(scenario_instance, args)

    if scenario_instance.settings.simulator_name == "MultiScaleSolver":
        generate_multiscale_inputs(scenario_instance)

    save_scenario(os.path.join(results_folder_path, "simulation_settings.json"), asdict(scenario_instance))

    return scenario_instance


if __name__ == '__main__':

    with open(os.path.join(default_settings_path, "scenario_default.json"), "r") as f: #
        default_scenario = scenario(**json.load(f))
    f.close()

    """ Define Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

    """ Parse Arguments  """
    input_parser = generate_input_parser(parent_parser=[generate_settings_parser(add_help = False),
                                                        generate_virtual_subjects_parser(add_help = False)]
                                        )
    
    args = input_parser.parse_args()

    settings_file = generate_simulation_settings_main(scenario_instance=default_scenario, args=args, results_folder_path=results_folder_path)

    settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

    settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

    if args.verbose:
        pprint.PrettyPrinter(indent=1).pprint(settings_file)
