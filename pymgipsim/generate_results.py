import pickle, pandas
import os

import pandas as pd

from pymgipsim.Utilities.paths import results_path
from pymgipsim.Utilities import simulation_folder
from pymgipsim.Utilities.units_conversions_constants import DEFAULT_RANDOM_SEED

from pymgipsim.InputGeneration.parsers import generate_input_parser
from pymgipsim.Settings.parser import generate_settings_parser
from pymgipsim.VirtualPatient.parser import generate_virtual_subjects_parser, generate_results_parser
from pymgipsim.VirtualPatient.VirtualPatient import VirtualCohort

from pymgipsim.generate_settings import generate_simulation_settings_main
from pymgipsim.generate_inputs import generate_inputs_main
from pymgipsim.generate_subjects import generate_virtual_subjects_main
from pymgipsim.Utilities.simulation_folder import save_to_xls
from pymgipsim.VirtualPatient import Models
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from py_agata.py_agata import Agata

import numpy as np


# Defines the random seed using the default seed
np.random.seed(DEFAULT_RANDOM_SEED)


def get_metrics(model):
	metrics = []
	analyzer = Agata()
	for patientidx in range(model.states.as_array.shape[0]):
		match model.name:
			case Models.T1DM.ExtHovorka.Model.name:
				glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(model.states.as_array[patientidx, model.glucose_state, :] /model.parameters.VG[patientidx])
			case Models.T1DM.IVP.Model.name:
				glucose = model.states.as_array[patientidx, model.glucose_state, :]
		metrics.append(analyzer.analyze_glucose_profile(pd.DataFrame({'t':pd.to_datetime(model.time.as_datetime).tz_localize(None),'glucose':glucose})))
	return metrics

def generate_results_main(scenario_instance, args, results_folder_path, faults_array=None):
    
	if not args['no_print']:
		print(f">>>>> Generating Model Results")

	cohort = VirtualCohort(scenario_instance)

	cohort.singlescale_model.preprocessing()

	match scenario_instance.settings.simulator_name:

		case 'MultiScaleSolver':
			cohort.multiscale_model.preprocessing()

			cohort.model_solver.do_simulation(no_progress_bar = args['no_progress_bar'])
			
			with open(os.path.join(results_folder_path, "model.pkl"), 'wb') as f:
				pickle.dump(cohort.model_solver.singlescale_model, f)

			with open(os.path.join(results_folder_path, "multiscale_model.pkl"), 'wb') as f:
				pickle.dump(cohort.model_solver.multiscale_model, f)

		case 'SingleScaleSolver':
			_ , faults_label = cohort.model_solver.do_simulation(no_progress_bar = args['no_progress_bar'], faults_array=faults_array)

			cohort.model_solver.model.faults_label = np.array(faults_label)

			model = cohort.model_solver.model

			with open(os.path.join(results_folder_path, "model.pkl"), 'wb') as f:
				pickle.dump(model, f)

			# [subjects x states x samples]
			state_results = cohort.model_solver.model.states.as_array

			# list [states]
			state_names = cohort.model_solver.model.states.state_names

			# list [states]
			state_units = cohort.model_solver.model.states.state_units

			# Save insulin input
            insulin_array = model.inputs.as_array[:, 3, :]  
            df_insulin = pd.DataFrame(insulin_array.T)      

            insulin_path = os.path.join(results_folder_path, "insulin_input.csv")
            df_insulin.to_csv(insulin_path, index=False)

			if args['to_excel']:
				if not args['no_print']:
					print(">>>>> Formatting and Saving Results")
				with pandas.ExcelWriter(os.path.join(results_folder_path, "model_state_results.xlsx")) as writer:
					save_to_xls(state_results, state_names, state_units, writer, args["no_progress_bar"], faults_label)

	return cohort #None#get_metrics(cohort.singlescale_model)

if __name__ == '__main__':

	""" Define Results Path """
	_, _, _, results_folder_path = simulation_folder.create_simulation_results_folder(results_path)


	""" Parse Arguments  """
	settings_parser = generate_settings_parser(add_help = False)

	cohort_parser = generate_virtual_subjects_parser(add_help=False)

	input_settings_parser = generate_input_parser(add_help=False)

	results_parser = generate_results_parser(parent_parser=[settings_parser, cohort_parser, input_settings_parser], add_help = True)

	args = results_parser.parse_args()

	settings_file = simulation_folder.load_settings_file(args, results_folder_path)

	if not args.defined_scenario:

		settings_file = generate_simulation_settings_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

		settings_file = generate_virtual_subjects_main(scenario_instance=settings_file, args=args, results_folder_path=results_folder_path)

		settings_file = generate_inputs_main(scenario_instance = settings_file, args = args, results_folder_path=results_folder_path)

	model,_ = generate_results_main(scenario_instance = settings_file, args = args, results_folder_path = results_folder_path)
