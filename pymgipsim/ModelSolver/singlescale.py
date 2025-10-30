from abc import ABC, abstractmethod

import numpy as np

from ..ODESolvers.ode_solvers import euler_single_step, rk4_single_step
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim import Controllers
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.VirtualPatient.Models import T1DM
from tqdm import tqdm

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
                        self.model.inputs.uInsulin.sampled_signal[:, 0] = (
                            self.controller.basal.sampled_signal[:, 0]
                        )
                    case T1DM.IVP.Model.name:
                        self.model.inputs.basal_insulin.sampled_signal[:, 0] = self.controller.basal.sampled_signal[:, 0]
                self.model.preprocessing()
            case Controllers.HCL0.controller.Controller.name:
                self.controller = Controllers.HCL0.controller.Controller(self.scenario_instance)
                self.model.inputs.uInsulin.sampled_signal[:, 0] = UnitConversion.insulin.Uhr_to_mUmin(np.asarray([x.demographic_info.basal_rate for x in self.controller.controllers]))
                self.model.preprocessing()
            case Controllers.OpenAPS.controller.Controller.name:
                self.controller = Controllers.OpenAPS.controller.Controller(
                    self.scenario_instance
                )
                self.model.inputs.uInsulin.sampled_signal[:, 0] = (
                    UnitConversion.insulin.Uhr_to_mUmin(
                        np.asarray(self.controller.demographic_info.basal)
                    )
                )
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

    def do_simulation(self, no_progress_bar):

        """ Initialize """
        state_results = self.model.states.as_array
        inputs = self.model.inputs.as_array
        parameters = self.model.parameters.as_array

        self.set_controller(self.scenario_instance.controller.name)

        state_results[:, :, 0] = self.model.initial_conditions.as_array
        for sample in tqdm(range(1, inputs.shape[2]), disable = no_progress_bar):

            self.controller.run(measurements=state_results[:, self.model.output_state, sample - 1], inputs=inputs, states=state_results, sample=sample-1)

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

        return state_results
