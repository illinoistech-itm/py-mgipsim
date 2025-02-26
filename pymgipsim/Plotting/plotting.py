import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
import pymgipsim.VirtualPatient.Models as Models


""" 
#######################################################################################################################
Model States
#######################################################################################################################
"""

def plot_subject_response(loaded_model, scenario:scenario, patientidx, faults_label=None):
    mpl.rcParams["font.size"] = 12
    fig = plt.figure(figsize = (12, 8))
    cho_color = [0.259, 0.125, 0.329]
    insulin_color = [0, 0.639, 0.224]
    snack_color = [0.7, 0.7, 0.7]
    glucose_color = [0.69, 0.235, 0.129]
    sglt2i_color = [0.902, 0.941, 0]
    heart_rate_color = [1, 0.498, 0.545]#[1, 0.325, 0.392]
    time_axis = loaded_model.time.as_unix/60.0
    arrow_width = 0.5
    arrow_height = 5

    
    glucose = np.zeros_like(loaded_model.states.as_array[:, loaded_model.glucose_state, :])

    match loaded_model.name:
        case Models.T1DM.ExtHovorka.Model.name:
            # Original BG that input in the patient model
            # glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(loaded_model.states.as_array[patientidx, loaded_model.glucose_state, :] / loaded_model.parameters.VG[patientidx])
            # Manipulated BG that input in the controller
            glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(loaded_model.states.as_array[patientidx, loaded_model.output_state, :])
        case Models.T1DM.IVP.Model.name:
            glucose = loaded_model.states.as_array[patientidx, loaded_model.glucose_state, :]

    glucose_unit = "mg/dL"



    
    try:
        hr_times = scenario.inputs.heart_rate.start_time[patientidx]
        hr_magnitudes = scenario.inputs.heart_rate.magnitude[patientidx]
        hr_times = np.asarray(hr_times) / 60
        hr_magnitudes = np.asarray(hr_magnitudes)
        hr_times = np.append(hr_times, time_axis[-1])
        hr_magnitudes = np.append(hr_magnitudes, hr_magnitudes[-1])
        plt.fill_between(hr_times, hr_magnitudes, color = heart_rate_color, label='Heart rate [BPM]', alpha=0.4)
    except:
        pass

    try:
        if scenario.controller.name == "OpenLoop" or scenario.controller.name == "StochasticOpenLoop":
            label = True
            for magnitude,start_time in zip(scenario.inputs.bolus_insulin.magnitude[patientidx], scenario.inputs.bolus_insulin.start_time[patientidx]):
                if label:
                    label = False
                    plt.arrow(start_time/60.0, 0, 0, magnitude,head_width=arrow_width,head_length=arrow_height, facecolor=insulin_color, edgecolor=insulin_color, label='Bolus [U]')
                else:
                    plt.arrow(start_time/60.0, 0, 0, magnitude, head_width=arrow_width, head_length=arrow_height,facecolor=insulin_color, edgecolor=insulin_color)
        else:
            match loaded_model.name:
                case Models.T1DM.ExtHovorka.Model.name:
                    plt.fill_between(time_axis, UnitConversion.insulin.mUmin_to_Uhr(loaded_model.inputs.as_array[patientidx, 3, :]), color=insulin_color, label='Insulin infusion rate [U/hr]')
                case Models.T1DM.IVP.Model.name:
                    plt.fill_between(time_axis, UnitConversion.insulin.uUmin_to_Uhr(loaded_model.inputs.as_array[patientidx, 1, :]),color=insulin_color, label='Insulin infusion rate [U/hr]')
    except:
        pass
    
    try:
        label = True
        for magnitude,start_time in zip(scenario.inputs.meal_carb.magnitude[patientidx], scenario.inputs.meal_carb.start_time[patientidx]):
            if label:
                plt.arrow(start_time/60.0, 0, 0, magnitude,head_width=arrow_width, head_length=arrow_height, facecolor=cho_color, edgecolor=cho_color, label='Meal carb [g]')
                label = False
            else:
                plt.arrow(start_time/60.0, 0, 0, magnitude, head_width=arrow_width, head_length=arrow_height,facecolor=cho_color, edgecolor=cho_color)
    except:
        pass


    try:
        label = True
        for magnitude,start_time in zip(scenario.inputs.snack_carb.magnitude[patientidx], scenario.inputs.snack_carb.start_time[patientidx]):
            if label:
                label = False
                plt.arrow(start_time/60.0, 0, 0, magnitude,head_width=arrow_width,head_length=arrow_height, facecolor=snack_color, edgecolor=snack_color, label='Snack carb [g]')
            else:
                plt.arrow(start_time/60.0, 0, 0, magnitude, head_width=arrow_width,head_length=arrow_height, facecolor=snack_color, edgecolor=snack_color)
    except:
        pass
    try:
        label = True
        for magnitude,start_time in zip(scenario.inputs.sgl2i.magnitude[patientidx], scenario.inputs.sgl2i.start_time[patientidx]):
            if magnitude>0.0:
                if label:
                    label = False
                    plt.arrow(start_time/60.0, 0, 0, magnitude,head_width=arrow_width,head_length=arrow_height, facecolor=sglt2i_color, edgecolor=sglt2i_color, label='SGLT2i [mg]')
                else:
                    plt.arrow(start_time/60.0, 0, 0, magnitude, head_width=arrow_width, head_length=arrow_height,facecolor=sglt2i_color, edgecolor=sglt2i_color)
    except:
        pass

    # plot faulty area
    try:
        in_fault_region = False
        start_minute = 0
        # Use a flag to ensure the legend label is only added once
        label_added = False
        for i, label_val in enumerate(faults_label):
            is_fault = (label_val != 'None')

            # Check for the beginning of a new faulty region
            if is_fault and not in_fault_region:
                in_fault_region = True
                start_minute = i

            # Check for the end of a faulty region
            elif not is_fault and in_fault_region:
                in_fault_region = False
                end_minute = i  # The region ends at the current minute

                # Plot the extracted start and end points
                current_label = 'Faulty region' if not label_added else None
                plt.axvspan(start_minute / 60.0, end_minute / 60.0,
                           color='gray', alpha=0.15, label=current_label)
                label_added = True
    except:
        pass

    plt.plot(time_axis, glucose, color=glucose_color, label="Blood glucose ["+glucose_unit+"]")
    plt.grid()
    # plt.ylim((0, 400))
    plt.xlabel('Time [h]')
    plt.ylabel('Magnitude')
    plt.xlim([0, time_axis[-1]])
    if np.max(glucose) < 250.0:
        plt.ylim([0.0, 250.0])
    else:
        plt.ylim([0.0, np.nanmax(glucose)])
    try:
        plt.title(scenario.patient.model.name+" "+scenario.patient.files[patientidx].replace(".json",""))
    except:
        pass
    plt.legend(loc='upper right')

    return fig


def plot_bgc(time, glucose, figsize, figcolor):
    """
    Plot the blood glucose concentration (BGC) over time.

    Parameters:
    - formatted_time: np.ndarray
        Time array for plotting.
    - formatted_glucose: np.ndarray
        2D array containing blood glucose data for different subjects.

    Returns:
    - fig: matplotlib.figure.Figure
        The created matplotlib figure.


    """

    mpl.rcParams["font.size"] = 18
    fig = plt.figure(figsize = figsize)

    glucose_mean = glucose.mean(axis=0)
    glucose_std = glucose.std(axis=0)

    time = time/60.0

    plt.plot(time, glucose_mean, color=figcolor, linewidth=1.8)#[0.961*0.5, 0.867*0.5, 0.208*0.5]
    plt.fill_between(time, glucose_mean - glucose_std, glucose_mean + glucose_std, alpha=0.4, color=figcolor)#[0.961, 0.867, 0.208],label='Without SGL2Ti'
    plt.legend()
    plt.grid()
    plt.title("Blood glucose response")
    plt.xlabel('Time [h]')
    plt.ylabel('Blood glucose [mg/dL]')
    plt.ylim([50.0, 250.0])
    plt.xlim([0.0, time[-1]])

    return fig


def plot_all_states(time, all_states, state_units, state_names, figsize, figcolor):
    """
    Plot all states over time in subplots.

    Parameters:
    - formatted_time: np.ndarray
        Time array for plotting.
    - all_states: np.ndarray
        4D array containing state data for different days and variables.
    - state_names: list
        List of state variable names.
    - state_units: list
        List of units corresponding to state variables.

    Returns:
    - fig: matplotlib.figure.Figure
        The created matplotlib figure.
    """
    mpl.rcParams["font.size"] = 10
    rows = int(np.sqrt(all_states.shape[1]))
    cols = rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    state_num = 0
    for row_num in range(rows):
        for col_num in range(cols):
            mean = all_states[:, state_num, :].mean(axis=0)
            std = all_states[:, state_num, :].std(axis=0)

            axes[row_num, col_num].plot(time, mean, color=figcolor)
            axes[row_num, col_num].fill_between(time, mean - std, mean + std, alpha=0.4, color=figcolor)
            axes[row_num, col_num].set_xlabel('Time (h)')
            axes[row_num, col_num].set_ylabel(state_units[state_num])
            axes[row_num, col_num].set_title(state_names[state_num])

            state_num += 1

    plt.xlabel('Time (h)')

    plt.tight_layout()

    return fig


def plot_bw(time, bw_state, bw_unit, state_name, figsize, figcolor):

    fig = plt.figure(figsize = figsize)

    mean = bw_state.mean(axis = 0)[0]
    std = bw_state.std(axis = 0)[0]

    plt.plot(time, mean, color=figcolor)
    plt.fill_between(time, mean - std, mean + std, alpha=0.4, color=figcolor)

    plt.xlabel('Days')
    plt.ylabel(f'{state_name} ({bw_unit})')
    plt.ylim((0.9 * np.min(mean - std), 1.1 * np.max(mean + std)))

    return fig


""" 
#######################################################################################################################
Inputs
#######################################################################################################################
"""

def plot_input_signals(time, input_array, input_names):

    num_inputs = input_array.shape[1]

    if num_inputs > 1:
        num_rows = np.ceil(np.sqrt(num_inputs)).astype(int)
        num_cols = num_rows.astype(int)

    fig, axes = plt.subplots(num_rows, num_cols)

    input_num = 0
    for row in range(num_rows):
        for col in range(num_cols):

            if input_num < num_inputs:

                mean = input_array[:, input_num, :].mean(axis = 0)
                std = input_array[:, input_num, :].std(axis = 0)

                axes[row, col].plot(time, mean)
                axes[row, col].fill_between(time, mean - std, mean + std, alpha = 0.4)
                axes[row, col].set_xlabel('Time (min)')
                axes[row, col].set_title(f"{input_names[input_num]}")
                axes[row, col].set_ylim([0, 1.1*max(mean + std)])

            input_num += 1

    fig.tight_layout()

    return fig
