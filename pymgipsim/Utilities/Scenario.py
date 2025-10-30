from dataclasses import dataclass
from pymgipsim.InputGeneration.signal import Events
import sys, json, numpy

@dataclass(slots=True)
class settings():
    """ General, simulator wide settings.

            Mirrors the settings field of the scenario JSON file.

            Parameters:
                sampling_time : Sampling time of the simulation [min].
                solver_name : ODE solver name (Euler or RK4)
                number_of_subjects : Cohort simulation size.
                start_time : Start time of the simulation in %d-%m-%Y %H:%M:%S datetime format.
                end_time : End time of the simulation in %d-%m-%Y %H:%M:%S datetime format.
                simulator_name : Currently openloop single scale, will be extended with more capabilities.

    """
    sampling_time: int
    simulator_name: str
    solver_name: str
    save_directory: str
    start_time: int
    end_time: int
    random_seed: int
    random_state: dict

@dataclass(slots=True)
class controller():
    name: str
    parameters: list


@dataclass(slots=True)
class model():
    """ Model descriptor.

            Mirrors the model field of the scenario JSON file.

            Parameters:
                name : Name of the model (T1DM.Hovorka, T1DM.IVP, T2DM.Jauslin)
                parameters : Array of model specific parameter values.
                initial_conditions : Array of initial states for the simulation

    """
    name: str = None
    parameters: list = None
    initial_conditions: list = None

@dataclass(slots=True)
class mscale():
    """ Model descriptor.

            Mirrors the model field of the scenario JSON file.

            Parameters:
                name : Name of the model (T1DM.Hovorka, T1DM.IVP, T2DM.Jauslin)
                parameters : Array of model specific parameter values.
                initial_conditions : Array of initial states for the simulation

    """
    models: list = None
    parameters: list = None



@dataclass(slots=True)
class demographic_info():
    """ Model independent patient information.

            Mirrors the demographic info field of the scenario JSON file.

            Parameters:
                body_weight : Body weights of the patients [kg].
                egfr : Glomural filtration rates [mL/min/1.73 m^2 BSA]
                basal : Basal insulin rates [U/hr].
                height : Height [m].
                total_daily_basal : [U]

    """
    body_weight_range: list = None
    body_weight: list = None

    renal_function_category: list = None
    egfr: list = None
    basal: list = None
    height: list = None
    total_daily_basal: list = None
    carb_insulin_ratio: list = None
    resting_heart_rate: list = None
    correction_bolus: list = None
    HbA1c: list = None
    waist_size: list = None


    baseline_daily_energy_intake: list = None
    baseline_daily_energy_expenditure: list = None
    baseline_daily_urinary_glucose_excretion: list = None

@dataclass(slots=True)
class patient():
    """ Describes the patient/cohort.

            Mirrors the patient field of the scenario JSON file.

            Parameters:
                demographic_info (demographic_info) : Stores the model independent patient information.
                model (model) : Model descriptor.

    """
    demographic_info: demographic_info = None
    model: model = None
    mscale: mscale = None
    files: list = None
    number_of_subjects: int = None

    def __post_init__(self):
        scenario_module = sys.modules[__name__]
        # Casts imported dictionaries to corresponding dataclasses
        for attribute in self.__slots__:
            if not (attribute == 'files' or attribute == 'number_of_subjects'):
                try:
                    setattr(self, attribute, getattr(scenario_module, attribute)(**getattr(self, attribute)))
                except:
                    setattr(self, attribute, None)
                    # print("Loaded scenario lacks: " + attribute + " information.")



@dataclass(slots=True)
class inputs():
    """ Stores all the possible inputs to all the possible models.

            Mirrors the patient field of the scenario JSON file.

            Note:
                Undefined/ not required inputs for a specific model are None.

            Parameters:
                meal_carb (Events) : Carb content of the meals [g].
                snack_carb (Events) : Carb content of the snacks [g].
                sgl2i (Events) : SGL2i drug intakes [mg].
                basal_insulin (Events) : Basal insulin rates [U/hr].
                bolus_insulin (Events) : Bolus insulin intakes [U].
                iob (Events) : Insulin On Board tracking [U] - populated by controller during simulation.
                heart_rate (Events) : Heart rate values [BPM].
                taud (Events) : Meal carb absorption times [min].

    """
    meal_carb: Events = None
    snack_carb: Events = None

    sgl2i: Events = None
    basal_insulin: Events = None
    bolus_insulin: Events = None
    bodyweighteffect: Events = None
    iob: Events = None

    heart_rate: Events = None
    taud: Events = None
    running_speed: Events = None
    running_incline: Events = None
    cycling_power: Events = None
    METACSM: Events = None
    energy_expenditure: Events = None

    daily_energy_intake: Events = None
    daily_energy_expenditure: Events = None
    daily_urinary_glucose_excretion: Events = None

    def __post_init__(self):
        # Casts imported dictionaries to Events classes
        for attribute in self.__slots__:
            try:
                setattr(self, attribute, Events(**getattr(self, attribute)).as_dict())
            except:
                setattr(self, attribute, None)
                # print("Loaded inputs lacks: " + attribute + " information.")


@dataclass(slots=True)
class input_generation():

    fraction_cho_intake: list = None
    fraction_cho_as_snack: list = None
    net_calorie_balance: list = None
    daily_energy_intake: list = None

    meal_duration: list = None
    snack_duration: list = None

    breakfast_time_range: list = None
    lunch_time_range: list = None
    dinner_time_range: list = None

    total_carb_range: list = None

    am_snack_time_range: list = None
    pm_snack_time_range: list = None

    sglt2i_dose_magnitude: list = None
    sglt2i_dose_time_range: list = None

    breakfast_carb_range: list = None
    lunch_carb_range: list = None
    dinner_carb_range: list = None

    am_snack_carb_range: list = None
    pm_snack_carb_range: list = None

    running_start_time: list = None
    running_duration: list = None
    running_incline: list = None
    running_speed: list = None
    cycling_start_time: list = None
    cycling_duration: list = None
    cycling_power: list = None


@dataclass(slots=True)
class scenario():
    """ Stores all the necessary information to uniquely define a simulation.

            Mirrors the scenario JSON file.

            Note:
                Undefined/ not required field are None.

            Parameters:
                settings (settings) : General, simulator wide settings.
                input_generation (input_generation) : Defines parameters for random input generation.
                inputs (inputs) : Defines the events (start time, magnitude and duration) of specific inputs.
                patient (patient) : Describes the simulated virtual cohort.

    """
    settings: settings
    input_generation: input_generation
    inputs: inputs
    patient: patient
    controller: controller

    def __post_init__(self):
        scenario_module = sys.modules[__name__]
        # Casts imported dictionaries to corresponding dataclasses
        for attribute in self.__slots__:

            try:
                setattr(self, attribute, getattr(scenario_module, attribute)(**getattr(self, attribute)))
            except:
                setattr(self, attribute, None)
                # print("Loaded scenario lacks: " + attribute + " information.")


def load_scenario(path):

    with open(path, "r") as f: #
        loaded_scenario = json.load(f)
    f.close()

    scenario_instance = scenario(**loaded_scenario)

    return scenario_instance
    
def save_scenario(path, scenario):
    with open(path, "w") as f:
        json.dump(scenario, f, indent=4)
