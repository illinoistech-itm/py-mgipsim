import argparse
import pandas as pd
from pymgipsim.Interface.Messages.parser_colors import color_help_text, color_group_header_text

def generate_faults_parser(add_help=False):
    """
    Creates a dedicated parser for fault injection and random scenario arguments.
    """
    parser = argparse.ArgumentParser(add_help=add_help)

    # Use an argument group for clean help messages
    fault_group = parser.add_argument_group(color_group_header_text('Faults and Scenario Randomization'))

    # Mutually exclusive group for fault modes
    fault_mode = fault_group.add_mutually_exclusive_group()
    fault_mode.add_argument(
        '--faults_file',
        type=str,
        default=None,
        help=color_help_text('Path to a faults specification CSV file with .')
    )

    # fault_group.add_argument(
    #     '-simulation_start_time',
    #     type=pd.Timestamp,
    #     default=pd.Timestamp('2025-10-01 00:00:00'),
    #     help="Simulation start time in a format like 'YYYY-MM-DD HH:MM:SS'."
    # )

    fault_group.add_argument(
        '--random_fault_intensity',
        type=float,
        default=None,
        help=color_help_text('Intensity(Ratio) for random fault generation (0.0 to 1.0).')
    )

    fault_group.add_argument(
        '--fault_type',
        nargs='+',
        default=['max_basal', 'min_basal', 'positive_basal', 'negative_basal', 'unknown_stop',
                 'unknown_under', 'missing_signal', 'positive_spike', 'negative_spike', 'negative_bias',
                 'positive_bias', 'min_reading', 'max_reading', 'repeated_reading', 'false_meal',
                 'false_bolus', 'repeated episode'],
        choices=['max_basal', 'min_basal', 'positive_basal', 'negative_basal', 'unknown_stop',
                 'unknown_under', 'missing_signal', 'positive_spike', 'negative_spike', 'negative_bias',
                 'positive_bias', 'min_reading', 'max_reading', 'repeated_reading', 'false_meal',
                 'false_bolus', 'repeated episode'],
        help=color_help_text('Select types of faulty injection.')
    )


    fault_group.add_argument(
        '--random_scenario',
        nargs='+',
        default=None,
        choices=['meal_carb', 'meal_start_time', 'snack_carb', 'snack_start_time',
                 'cycling_power', 'cycling_start_time', 'cycling_duration',
                 'running_speed', 'running_start_time', 'running_duration'],
        help=color_help_text('List of targets for scenario randomization.')
    )

    fault_group.add_argument(
        '--random_scenario_methods',
        nargs='+',
        default=['heavy', 'light', 'early', 'delayed', 'skipped'],
        choices=['heavy', 'light', 'early', 'delayed', 'skipped'],
        help=color_help_text("List of methods for scenario randomization.\n heavy - Increase 10-40 percent of magnitude or duration.\n light - Decrease 10-40 percent of magnitude or duration.\n early - 1-2 hour before original start_time.\n delayed - 1-2 hour after original start_time.\n skipped - Set 0 of magnitude")
    )


    fault_group.add_argument(
        '--random_scenario_intensity',
        type=float,
        default=0.1,
        help=color_help_text('Ratio for scenario randomization.')
    )

    return parser