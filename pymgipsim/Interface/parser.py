import argparse
from pymgipsim.Interface.Messages.parser_colors import *
from pymgipsim.Interface.Messages.cli_directions import *

from pymgipsim.Settings.parser import generate_settings_parser
from pymgipsim.VirtualPatient.parser import generate_virtual_subjects_parser, generate_results_parser
from pymgipsim.InputGeneration.parsers import generate_input_parser, generate_carb_settings_parser, generate_activity_parser
from pymgipsim.Plotting.parser import generate_plot_parser, generate_plot_parser_multiscale
from pymgipsim.Utilities.parser import generate_load_parser
from pymgipsim.Controllers.parser import generate_controller_settings_parser

from pymgipsim.faultsGeneration.parser import generate_faults_parser

"""
#####################
Directions
#####################
"""

directions_parser = argparse.ArgumentParser(prog = 'Directions',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    exit_on_error = False,
                                    add_help = False
                                )


directions_parser.add_argument('-v', '--v', dest = 'directions_verbose', help = color_help_text('Display all directions'), action = 'store_true')

directions_parser.add_argument('-pa', '--parser', dest = 'directions_parser', help = color_help_text('Directions for parsing arguments from command line'), action = 'store_true')

directions_parser.add_argument('-c', '--cmd', dest = 'directions_cmd', help = color_help_text('Directions for working with this interface'), action = 'store_true')

directions_parser.add_argument('-m', '--model', dest = 'directions_model', help = color_help_text('Directions for choosing a model'), action = 'store_true')

directions_parser.add_argument('-in', '--inputs', dest = 'directions_inputs', help = color_help_text('Directions for settings different inputs'), action = 'store_true')

directions_parser.add_argument('-s', '--simulate', dest = 'directions_simulate', help = color_help_text('Directions for running a simulation'), action = 'store_true')

directions_parser.add_argument('-pl', '--plot', dest = 'directions_plot', help = color_help_text('Directions for plotting results'), action = 'store_true')


"""
#####################
CLI Parser
#####################
"""

def generate_parser_cli():

	parent_parser = [generate_load_parser(add_help = False),
					 generate_controller_settings_parser(add_help = False),
								generate_settings_parser(add_help = False),
								generate_virtual_subjects_parser(add_help = False),
								generate_input_parser(add_help = False),
								generate_results_parser(add_help = False),
								generate_plot_parser(add_help = False),
					 			generate_faults_parser(add_help=False)
								]
	parser = argparse.ArgumentParser(prog = initial_directions, parents = parent_parser, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	return parser