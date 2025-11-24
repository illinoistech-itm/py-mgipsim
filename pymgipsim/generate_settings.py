import argparse
import numpy as np
import os
from pymgipsim.Utilities.paths import default_settings_path, results_path
from pymgipsim.Settings.parser import generate_settings_parser
from pymgipsim.Utilities import simulation_folder
from pymgipsim.Utilities.Scenario import scenario, load_scenario, save_scenario
from pymgipsim.Utilities.Timestamp import Timestamp
from dataclasses import asdict
# Defines the random seed using the default seed

def generate_simulation_settings_main(scenario_instance: scenario, args: argparse.Namespace, results_folder_path: str) -> scenario:

    """
    This is used to generate the simulation settings.
    """
    if not args.no_print:
        print(f">>>>> Generating Simulation Settings")

    # pbar = tqdm.tqdm(total = 2, disable = args.no_progress_bar)

    end_time = Timestamp()
    end_time.as_unix = args.number_of_days * 24 * 60

    scenario_instance.controller.name = args.controller_name
    print("Current controller name: %s" % scenario_instance.controller.name)

    scenario_instance.settings.sampling_time = args.sampling_time
    scenario_instance.settings.start_time = 0
    scenario_instance.settings.end_time = end_time.as_unix
    scenario_instance.settings.random_seed = args.random_seed
    scenario_instance.settings.random_state = np.random.default_rng(args.random_seed).bit_generator.state
    if args.multi_scale:
        scenario_instance.settings.simulator_name = "MultiScaleSolver"
    else:
        scenario_instance.settings.simulator_name = "SingleScaleSolver"

    save_scenario(os.path.join(results_folder_path, "simulation_settings.json"), asdict(scenario_instance))

    return scenario_instance

if __name__ == '__main__':

    default_scenario = load_scenario(os.path.join(default_settings_path, "scenario_default.json"))

    """ Define Results Path """
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)

    """ Parse Arguments  """
    parser = generate_settings_parser(add_help = True)
    args = parser.parse_args()

    scenario_instance = generate_simulation_settings_main(args = args, scenario_instance = default_scenario, results_folder_path = results_folder_path)