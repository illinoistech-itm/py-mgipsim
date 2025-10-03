import argparse
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


def main(argv=None):
    """ Main execution function for the simulation pipeline. """

    '''Create and Parse Arguments'''
    args = generate_parser_cli().parse_args(argv)
    print("Arguments parsed successfully.")

    '''Initialization & Setup'''
    subprocess.run(['python', 'initialization.py'], check=True)
    _, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)
    settings_file = simulation_folder.load_settings_file(args, results_folder_path)
    activity_args_to_scenario(settings_file, args)

    '''Configure Faults and Random Scenarios'''
    faults_input = None
    random_scenario = None

    # Check for Fault Generation Options
    if args.faults_file:
        print(f"   - Loading faults from file: {args.faults_file}")
        faults_input = generate_faults.generate_faults_from_file(
            faults_file=args.faults_file,
            simulation_days=int(args.number_of_days)
        )
    elif args.random_fault_intensity:
        print(f"   - Generating random faults with intensity: {args.random_fault_intensity}")
        faults_input = generate_faults.generate_random_faults(
            simulation_days=int(args.number_of_days),
            intensity=args.random_fault_intensity,
            random_state=args.random_seed,
            faulty_type=args.fault_type
        )

    # Check for random scenario option
    if args.random_scenario:
        print(f"   - Using random scenario variations on inputs.")
        random_scenario = {
            'target': args.random_scenario,
            'method': args.random_scenario_methods,
            'intensity': args.random_scenario_intensity
        }

    '''RUN THE GENERATION PIPELINE'''
    if not args.scenario_name:
        print("\n Generating new scenario from provided arguments...")

        settings_file = generate_simulation_settings_main(
            scenario_instance=settings_file, args=args, results_folder_path=results_folder_path
        )
        settings_file = generate_virtual_subjects_main(
            scenario_instance=settings_file, args=args, results_folder_path=results_folder_path
        )
        settings_file = generate_inputs_main(
            scenario_instance=settings_file, args=args, results_folder_path=results_folder_path,
            random_scenario=random_scenario
        )
    else:
        print(f"\n Loading pre-defined scenario: {args.scenario_name}")

    model = generate_results_main(
        scenario_instance=settings_file, args=vars(args), results_folder_path=results_folder_path,
        faults_array=faults_input
    )

    figures = generate_plots_main(results_folder_path, args, faults_input)

    print(f"\n Simulation pipeline completed successfully! Results saved to {results_folder_path}")


if __name__ == '__main__':
    test_arguments = [
        '-pat', '0',
        '-d', '30',
        '-ns', '1',
        # '-ctrl', 'HCL0',
        # '-random_fault_intensity', '0.01',
        # '-fault_type', 'max_basal', 'positive_spike',
        '-random_scenario', 'meal_start_time',
        '-random_scenario_methods', 'early',
        # '-faults_file', 'pymgipsim/faultsGeneration/faults_specification.csv'
    ]
    main(test_arguments)
