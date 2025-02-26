from abc import ABC, abstractmethod

import numpy as np
import copy

from ..ODESolvers.ode_solvers import euler_single_step, rk4_single_step
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim import Controllers
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models import T1DM
from tqdm import tqdm

from pymgipsim.faultsGeneration.faults_injection import FaultsInjection
from pymgipsim.faultsGeneration.generate_faults import faults_id_dict
from pymgipsim.faultsGeneration import faults_utils

class SolverBase(ABC):


    def __init__(self, scenario_instance: scenario, model: BaseModel):

        # Directory where the results should be stored
        self.scenario_instance = scenario_instance

        self.model = model

        # ODE solver function
        self.set_solver(self.scenario_instance.settings.solver_name)

    def set_controller(self, controller_name):
        match controller_name:
            case Controllers.OpenLoop.controller.Controller.name:
                self.controller = Controllers.OpenLoop.controller.Controller()
            case Controllers.StochasticOpenLoop.controller.Controller.name:
                self.controller = Controllers.StochasticOpenLoop.controller.Controller(self.scenario_instance)
                self.model.inputs.uInsulin.sampled_signal[:,0] = self.controller.insulin.sampled_signal[:,0]
                self.model.preprocessing()
            case Controllers.SAPT.controller.Controller.name:
                converted_glucose = lambda glucose: UnitConversion.glucose.concentration_mgdl_to_mmolL(glucose) if self.model.states.state_units[self.model.output_state] == 'mmol/L' else glucose
                self.controller = Controllers.SAPT.controller.Controller(self.scenario_instance, converted_glucose(100.0), self.model.states.state_units)
                match self.controller.model_name:
                    case T1DM.ExtHovorka.Model.name:
                        self.model.inputs.uInsulin.sampled_signal[:,0] = self.controller.basal.sampled_signal[:,0]
                    case T1DM.IVP.Model.name:
                        self.model.inputs.basal_insulin.sampled_signal[:, 0] = self.controller.basal.sampled_signal[:, 0]
                self.model.preprocessing()
            case Controllers.HCL0.controller.Controller.name:
                self.controller = Controllers.HCL0.controller.Controller(self.scenario_instance)
                self.model.inputs.uInsulin.sampled_signal[:, 0] = UnitConversion.insulin.Uhr_to_mUmin(np.asarray([x.demographic_info.basal_rate for x in self.controller.controllers]))
                self.model.preprocessing()
            case _:  # Default case
                raise Exception("Undefined controller, Add it to the ModelSolver class.")


    def set_solver(self, solver_name):
        match solver_name:
            case "RK4":
                self.ode_solver = rk4_single_step
            case 'Euler':
                self.ode_solver = euler_single_step

    @abstractmethod
    def do_simulation(self):
        pass



class SingleScaleSolver(SolverBase):

    name = "SingleScaleSolver"

    def do_simulation(self, no_progress_bar, faults_array=None):

        """ Initialize """
        # Patient states. Not affected by the CGM readings manipulation but only by insulin delivery
        state_results = self.model.states.as_array
        inputs = self.model.inputs.as_array
        parameters = self.model.parameters.as_array

        self.set_controller(self.scenario_instance.controller.name)

        state_results[:, :, 0] = self.model.initial_conditions.as_array

        if faults_array is None:
            for sample in tqdm(range(1, inputs.shape[2]), disable=no_progress_bar):
                self.controller.run(measurements=state_results[:, self.model.output_state, sample - 1],
                                    inputs=inputs, states=state_results, sample=sample - 1)

                state_results[:, :, sample] = self.ode_solver(
                    f=self.model.model,
                    time=float(sample),
                    h=float(self.model.sampling_time),
                    initial=state_results[:, :, sample - 1].copy(),
                    parameters=parameters,
                    inputs=inputs[:, :, sample - 1]
                )

            self.model.states.as_array = state_results
            self.model.inputs.as_array = inputs

            return state_results, None

        else:
            """Initialize Fault Injection"""
            # Silicon states. Computed for the controller and will be affected when launching false CGM readings attack.
            silicon_state_results = copy.copy(self.model.states.as_array)
            faults_engine = FaultsInjection()
            faults_label = ['None'] * state_results.shape[-1]
            first_repeat = True
            # Get fault id type
            fault_label_dict = {name: id for id, name in faults_id_dict.items()}
            cgm_faults_ids = [fault_label_dict.get(name) for name in faults_engine.bg_faults]
            insulin_faults_ids = [fault_label_dict.get(name) for name in faults_engine.insulin_faults]
            print('Fault Injection Initialized')

            for sample in tqdm(range(1, inputs.shape[2]), disable = no_progress_bar):
                # Inject false state: meal, bolus
                # if (sample in range(600, 600)) & (self.scenario_instance.controller.name == 'HCL0'):
                #     # input state: uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure
                #     # # uSlowCarbs
                #     # false_carb, f_label = faults_engine_i.false_meal(inputs[:, 1, :sample+1])
                #     # faults_label[sample] = f_label
                #     # inputs[:, 1, sample] = false_carb.copy()
                #     # uInsulin
                #     false_bolus, f_label = faults_engine_i.false_bolus()
                #     faults_label[sample] = f_label
                #     inputs[:, 3, sample] = false_bolus.copy()

                # inject CGM readings' faults before input into controller.
                if faults_array[sample - 1] in cgm_faults_ids:
                    f_label = faults_id_dict[faults_array[sample-1]]

                    true_last_bg = copy.copy(state_results[:, self.model.output_state, sample - 1])
                    silicon_state_results[:, :, sample - 1] = self.ode_solver(
                        f=self.model.model,
                        time=float(sample - 1),
                        h=float(self.model.sampling_time),
                        initial=silicon_state_results[:, :, sample - 2].copy(),
                        parameters=parameters,
                        inputs=current_input  # inputs[:, :, sample - 1]
                    )

                    if f_label == 'repeated_episode':
                        if first_repeat:
                            meal_episode_start = faults_utils.get_first_meal_index(carb_past=inputs[:, 1, :sample])
                            episode_dist = sample - 1 - meal_episode_start
                            first_repeat = False
                        false_last_bg = faults_engine.run_cgm(f_label, bg=true_last_bg, bg_dist=episode_dist, bg_past=state_results[:, self.model.output_state, :sample])
                    elif f_label == 'repeated_reading':
                        false_last_bg = faults_engine.run_cgm(fault_type=f_label, bg=true_last_bg, bg_past=silicon_state_results[:, self.model.output_state, sample - 2])
                    else:
                        false_last_bg = faults_engine.run_cgm(fault_type=f_label, bg=true_last_bg)

                    faults_label[sample-1] = f_label
                    silicon_state_results[:, self.model.output_state, sample-1] = false_last_bg.copy()

                    # Keep real readings for the patient model and fake readings for the controller and display
                    self.controller.run(measurements=false_last_bg, inputs=inputs,
                                        states=silicon_state_results, sample=sample - 1)
                else:
                    silicon_state_results[:, :, sample - 1] = copy.copy(state_results[:, :, sample - 1])
                    self.controller.run(measurements=state_results[:, self.model.output_state, sample - 1],
                                        inputs=inputs, states=state_results, sample=sample-1)

                # if (sample in range(1000, 1200)) & (self.scenario_instance.controller.name == 'HCL0'):
                #     silicon_state_results[:, :, sample-1] = self.ode_solver(
                #         f=self.model.model,
                #         time=float(sample-1),
                #         h=float(self.model.sampling_time),
                #         initial=silicon_state_results[:, :, sample - 2].copy(),
                #         parameters=parameters,
                #         inputs=current_input  # inputs[:, :, sample - 1]
                #     )
                #     # repeat episode
                #     fault = 'repeated_episode'
                #     if (fault == 'repeated_episode') & first_repeat:
                #         first_repeat = False
                #         false_episode, f_label = faults_engine_bg.repeated_episode(
                #             bg=state_results[:, self.model.output_state, :sample], carb_past=inputs[:, 1, :sample], copy_len=200)
                #
                #     false_last_bg = false_episode[:, sample-1].copy()
                #     # Manipulate CGM readings
                #     # true_last_bg = copy.copy(state_results[:, self.model.output_state, sample-1])
                #     # false_last_bg, f_label = faults_engine_bg.zero_readings(bg=true_last_bg)
                #     faults_label[sample-1] = f_label
                #     silicon_state_results[:, self.model.output_state, sample-1] = false_last_bg.copy()
                #     # # Keep real readings for the patient model and fake readings for the controller and display
                #     self.controller.run(measurements=false_last_bg, inputs=inputs,
                #                         states=silicon_state_results, sample=sample - 1)
                # else:
                #     silicon_state_results[:, :, sample - 1] = copy.copy(state_results[:, :, sample - 1])
                #     self.controller.run(measurements=state_results[:, self.model.output_state, sample - 1],
                #                         inputs=inputs, states=state_results, sample=sample-1)

                # self.controller.run(measurements=state_results[:, self.model.output_state, sample - 1],
                #                     inputs=inputs, states=state_results, sample=sample - 1)


                # if (sample in range(600, 600)) & (self.scenario_instance.controller.name == 'HCL0'):
                #     # uInsulin
                #     false_bolus, f_label = faults_engine_i.false_bolus()
                #     faults_label[sample-1] = f_label
                #     inputs[:, 3, sample-1] = false_bolus

                # position 3 is the total insulin
                current_input = copy.copy(inputs[:, :, sample - 1])

                # inject controller faults before patient model.
                if faults_array[sample-1] in insulin_faults_ids:
                    f_label = faults_id_dict[faults_array[sample-1]]

                    if f_label == 'false_meal':
                        # input state: uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure
                        # uSlowCarbs
                        false_carb = faults_engine.run_insulin(f_label, carb_past=inputs[:, 1, :sample+1])
                        inputs[:, 1, sample] = false_carb.copy()
                    else:
                        f_basal = faults_engine.run_insulin(f_label, current_input[:, 3], sample)
                        current_input[:, 3] = f_basal

                    faults_label[sample - 1] = f_label
                    # When unknown malfunction happen with pump, the displayed controller activity will not accord with really injected dosage
                    if f_label not in ['unknown_stop', 'unknown_under', 'false_meal']:
                        inputs[:, 3, sample - 1] = f_basal

                # if (sample in range(720, 1200)) & (self.scenario_instance.controller.name == 'HCL0'):
                #     f_basal, f_label = faults_engine_i.negative_bias_basal(basal=current_input[:, 3])
                #     # position 3 is the total insulin in the current minute
                #     faults_label[sample-1] = f_label
                #     current_input[:, 3] = f_basal
                #     if f_label not in ['unknown_stop', 'unknown_under']:
                #         # when unknown malfunction happen, the present controller activity not accord with injected dosage
                #         inputs[:, 3, sample - 1] = f_basal

                state_results[:, :, sample] = self.ode_solver(
                    f=self.model.model,
                    time=float(sample),
                    h=float(self.model.sampling_time),
                    initial=state_results[:, :, sample - 1].copy(),
                    parameters=parameters,
                    inputs=current_input # inputs[:, :, sample - 1]
                )

            # Pass the CGM data received by the controller
            state_results[:, self.model.output_state, :] = silicon_state_results[:, self.model.output_state, :].copy()

            self.model.states.as_array = state_results
            self.model.inputs.as_array = inputs

            return state_results, faults_label
