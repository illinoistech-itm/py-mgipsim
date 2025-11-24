from pymgipsim.Plotting.plotting import *
import pickle
import json
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
import pymgipsim.VirtualPatient.Models as Models
import os


""" 
#######################################################################################################################
Model States
#######################################################################################################################
"""
def _generate_plot_subject(results_directory, faults_label=None, **kwargs):
    """
    Generate and save a detailed input-output plot for a single subject.

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - fig
    """

    with open(os.path.join(results_directory, "model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    with open(os.path.join(results_directory, "simulation_settings.json"), "r") as f:  #
        loaded_scenario = scenario(**json.load(f))
    f.close()

    patient = kwargs['plot_patient']

    fig = plot_subject_response(loaded_model, loaded_scenario, patient, faults_label=faults_label)
    fig.savefig(os.path.join(results_directory, "figures", "subject_" + fig.axes[0].get_title() + ".png"), bbox_inches='tight')

    return fig

def _generate_plot_bgc(results_directory, **kwargs):
    """
    Generate and save a blood glucose (BGC) plot for the simulation.

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - None
    """

    with open(os.path.join(results_directory, "model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    if 'figsize' in kwargs:
        figsize=tuple(kwargs['figsize'])
    else:
        figsize=(12,12)

    if 'color' in kwargs:
        figcolor = kwargs['color']
    else:
        figcolor = 'C0'

    glucose = np.zeros_like(loaded_model.states.as_array[:, loaded_model.glucose_state, :])


    match loaded_model.name:
        case Models.T1DM.ExtHovorka.Model.name:
            volume = loaded_model.parameters.VG

            for i in range(glucose.shape[0]):
                glucose[i] = UnitConversion.glucose.concentration_mmolL_to_mgdL(loaded_model.states.as_array[i, loaded_model.glucose_state, :] / volume[i])
                
        case Models.T1DM.IVP.Model.name:
            glucose = loaded_model.states.as_array[:, loaded_model.glucose_state, :]

    fig = plot_bgc(loaded_model.time.as_unix, glucose, figsize, figcolor)
    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, "figures", "bgc_plot.png"), bbox_inches='tight')

    return fig

def _generate_plot_all_states(results_directory, **kwargs):
    """
    Generate and save a plot for all states in the simulation.

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - None
    """

    if 'figsize' in kwargs:
        figsize=tuple(kwargs['figsize'])
    else:
        figsize=(12,12)

    if 'color' in kwargs:
        figcolor = kwargs['color']
    else:
        figcolor = 'C0'


    with open(os.path.join(results_directory, "model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    fig = plot_all_states(loaded_model.time.as_unix, loaded_model.states.as_array, loaded_model.states.state_units, loaded_model.states.state_names, figsize, figcolor)
    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, "figures", "all_states.png"), bbox_inches='tight')

    return fig


def _generate_bw(results_directory, **kwargs):
    """
    Generate and save a plot for body weight in a multiscale simulation

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - None
    """

    if 'figsize' in kwargs:
        figsize=tuple(kwargs['figsize'])
    else:
        figsize=(12,12)

    if 'color' in kwargs:
        figcolor = kwargs['color']
    else:
        figcolor = 'C0'


    with open(os.path.join(results_directory, "multiscale_model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    fig = plot_bw(loaded_model.time.as_unix, loaded_model.states.as_array, loaded_model.states.state_units, loaded_model.states.state_names, figsize, figcolor)
    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, "figures", "bodyweight.png"), bbox_inches='tight')

    return fig

""" 
#######################################################################################################################
Multiscale-Specific
#######################################################################################################################
"""


def _generate_bw(results_directory, **kwargs):
    """
    Generate and save a plot for body weight in a multiscale simulation

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - None
    """

    if 'figsize' in kwargs:
        figsize=tuple(kwargs['figsize'])
    else:
        figsize=(12,12)

    if 'color' in kwargs:
        figcolor = kwargs['color']
    else:
        figcolor = 'C0'


    with open(os.path.join(results_directory, "multiscale_model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    fig = plot_bw(loaded_model.time.as_unix, loaded_model.states.as_array, loaded_model.states.state_units, loaded_model.states.state_names, figsize, figcolor)
    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, "figures", "bodyweight.png"), bbox_inches='tight')

    return fig


""" 
#######################################################################################################################
Signals
#######################################################################################################################
"""

def _generate_input_signals(results_directory, **kwargs):
    """
    Generate and save a plot for body weight in a multiscale simulation

    Parameters:
    - simulation_folder: str
        Path to the simulation folder.

    Returns:
    - None
    """

    if 'figsize' in kwargs:
        figsize=tuple(kwargs['figsize'])
    else:
        figsize=(12,12)

    if 'color' in kwargs:
        figcolor = kwargs['color']
    else:
        figcolor = 'C0'


    with open(os.path.join(results_directory, "model.pkl"), 'rb') as f:
        loaded_model = pickle.load(f)

    fig = plot_input_signals(loaded_model.time.as_unix, loaded_model.inputs.as_array, [i for i in list(vars(loaded_model.inputs).keys()) if 'array' not in i])
    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, "figures", "input_singals.png"), bbox_inches='tight')

    return fig






""" 
#######################################################################################################################
Main
#######################################################################################################################
"""

def generate_plots_main(results_folder_path, args, faults_label=None):

    if not args.no_print:
        print(f">>>>> Generating Plots")

 
    plot_list = []

    bgc_fig = _generate_plot_bgc(results_folder_path, **vars(args))

    all_states_fig = _generate_plot_all_states(results_folder_path, **vars(args))

    input_signals_fig = _generate_input_signals(results_folder_path, **vars(args))

    plot_list = [bgc_fig, all_states_fig, input_signals_fig]

    if args.multi_scale:
        """ Body Weight Figure """
        bw_fig = _generate_bw(results_folder_path, **vars(args))
        plot_list.append(bw_fig)

        if not args.plot_body_weight and not args.plot_all:
            plt.close(bw_fig)
            plot_list.remove(bw_fig)

    if args.plot_patient is not None:
        subject_fig = _generate_plot_subject(results_folder_path, faults_label, **vars(args))
        plot_list.append(subject_fig)

    if not args.plot_blood_glucose and not args.plot_all:
        plt.close(bgc_fig)
        plot_list.remove(bgc_fig)

    if not args.plot_all_states and not args.plot_all:
        plt.close(all_states_fig)
        plot_list.remove(all_states_fig)

    if not args.plot_input_signals and not args.plot_all:
        plt.close(input_signals_fig)
        plot_list.remove(input_signals_fig)
            
    return plot_list

if __name__ == '__main__':
    pass
