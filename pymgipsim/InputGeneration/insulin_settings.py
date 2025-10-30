import numpy as np

from pymgipsim.InputGeneration.signal import Signal, Events
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Controllers import OpenLoop

def generate_basal_insulin(scenario_instance: scenario, args):
    basal = np.expand_dims(np.asarray(scenario_instance.patient.demographic_info.basal),1)
    start_time = np.ones_like(basal)*scenario_instance.settings.start_time
    if scenario_instance.controller.name == OpenLoop.controller.Controller.name:
        basal = args.basal_multiplier * basal
        return Events(start_time=start_time, magnitude=basal).as_dict()
    else:
        return Events(start_time=np.zeros_like(start_time), magnitude=np.zeros_like(basal)).as_dict()

def generate_bolus_insulin(scenario_instance: scenario, args):
    meal_times = np.asarray(scenario_instance.inputs.meal_carb.start_time)
    meal_durations = np.asarray(scenario_instance.inputs.meal_carb.duration)
    meal_magnitudes = np.asarray(scenario_instance.inputs.meal_carb.magnitude)

    if scenario_instance.controller.name == OpenLoop.controller.Controller.name:
        carb_insulin_ratio = np.expand_dims(np.asarray(scenario_instance.patient.demographic_info.carb_insulin_ratio),1)
        bolus_magnitudes = args.bolus_multiplier * np.divide(meal_magnitudes, carb_insulin_ratio)
        return Events(start_time= meal_times, duration=np.ones_like(meal_durations),
                           magnitude=bolus_magnitudes).as_dict()
    else:
        return Events(start_time= meal_times,
                    duration=np.ones_like(meal_durations),
                    magnitude=np.zeros_like(meal_magnitudes)).as_dict()

def generate_iob(scenario_instance: scenario, args):
    """Generate Insulin On Board (IOB) signal initialized to zeros.

    IOB will be populated by the controller during simulation.

    Args:
        scenario_instance: The scenario instance
        args: Command line arguments

    Returns:
        Events object with IOB initialized to zeros
    """
    # Initialize with a single zero event at start time for all patients
    num_patients = scenario_instance.patient.number_of_subjects
    start_time = np.ones((num_patients, 1)) * scenario_instance.settings.start_time
    magnitude = np.zeros((num_patients, 1))

    return Events(start_time=start_time, magnitude=magnitude).as_dict()
