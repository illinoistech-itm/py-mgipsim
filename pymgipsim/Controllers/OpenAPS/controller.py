import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import namedtuple
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.VirtualPatient import Models
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.Controllers.OpenAPS.Oref0 import ORefZeroController
from pymgipsim.VirtualPatient.Models import T1DM


# Create a namedtuple to match the expected observation format
CtrlObservation = namedtuple("CtrlObservation", ["CGM"])


class Controller:
    name = "OpenAPS"

    def __init__(self, scenario_instance: scenario):
        if scenario_instance.patient.model.name != T1DM.ExtHovorka.Model.name:
            raise Exception("OpenAPS controller only supports the ExtHovorka model.")

        self.dT = int(scenario_instance.settings.sampling_time)  # minutes
        self.controller = ORefZeroController()

        # Check server health using first controller
        if not self.controller.health_check():
            raise ConnectionError("Failed to connect to OpenAPS server")

        model_parameters = T1DM.ExtHovorka.Parameters(
            np.asarray(scenario_instance.patient.model.parameters)
        )
        self.demographic_info = scenario_instance.patient.demographic_info

        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            patient_name = self._patient_name(patient_idx)

            # Get basal rate from demographic info if available
            if hasattr(scenario_instance.patient.demographic_info, "basal"):
                basal_rate = self.demographic_info.basal[patient_idx]
            else:
                basal_rate = 0.7  # Default U/h

            # Get insulin sensitivity and carb ratio from demographic info if available
            if hasattr(self.demographic_info, "correction_bolus"):
                isf = self.demographic_info.correction_bolus[patient_idx]
            else:
                isf = 50  # Default mg/dL per U

            if hasattr(self.demographic_info, "carb_insulin_ratio"):
                carb_ratio = self.demographic_info.carb_insulin_ratio[patient_idx]
            else:
                carb_ratio = 10  # Default g/U

            if hasattr(self.demographic_info, "total_daily_basal"):
                max_daily_basal = self.demographic_info.total_daily_basal[patient_idx]
            else:
                max_daily_basal = 36  # Default U/day

            # Create profile for this patient
            profile = {
                "current_basal": basal_rate,  # U/h from demographic info
                "sens": isf,  # Insulin Sensitivity Factor (ISF) from demographic info
                "dia": 6,  # Duration of Insulin Action in hours
                "carb_ratio": carb_ratio,  # Carb Ratio (g/U) from demographic info
                "max_iob": 6,  # Maximum insulin on board allowed
                "max_basal": max(
                    3.5, basal_rate * 3
                ),  # Maximum temporary basal rate in U/h (3x normal)
                "max_daily_basal": max_daily_basal,  # Maximum daily basal insulin in U/day
                "max_bg": 140,  # Upper target
                "min_bg": 90,  # Lower target
                "maxCOB": 180,  # Maximum carbs on board
                "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": isf}]},
                "min_5m_carbimpact": 12.0,  # Minimum carb absorption rate
                "type": "current",  # Profile type
            }

            print(
                f"Initialized OpenAPS for {patient_name}"
                f" with basal {profile['current_basal']} U/h"
                f", isf {profile['sens']} mg/dL per U"
                f", carb ratio {profile['carb_ratio']} g/U"
                f", max basal {profile['max_basal']} U/h"
                f", and max daily basal {profile['max_daily_basal']} U/day"
            )
            # Create a controller instance for this patient
            self.controller.initialize_patient(patient_name, profile)

        # Track simulation time
        self.simulation_time = datetime.now()

    @staticmethod
    def _patient_name(index: int) -> str:
        return f"patient_{index}"

    def run(self, measurements, inputs, states, sample):
        if sample % self.dT == 0:
            # Update simulation time (convert sample to minutes)
            current_sim_time = self.simulation_time + timedelta(minutes=sample)

            for patient_idx in range(inputs.shape[0]):
                # Select hardcoded 1st patient (MPC currently works for single patient)
                glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(
                    measurements[patient_idx]
                )
                uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure, uIOB = (
                    inputs[patient_idx]
                )
                sum_meals = np.sum(
                    uFastCarbs[sample - self.dT + 1 : sample : sample + 1]
                    + uSlowCarbs[sample - self.dT + 1 : sample : sample + 1]
                )

                # Create observation for the controller
                observation = CtrlObservation(CGM=glucose)

                # Get insulin recommendation from server
                try:
                    action = self.controller.policy(
                        observation=observation,
                        reward=0,  # Not used by OpenAPS
                        done=False,  # Not used by OpenAPS
                        patient_name=self._patient_name(patient_idx),
                        meal=sum_meals,
                        time=current_sim_time,
                    )
                    basal = action.get("basal", 0.0) * 60  # U/min * 60 = U/h
                    bolus = action.get("bolus", 0.0)  # U
                    iob = action.get("iob", 0.0)  # U

                    insulin_rate = (
                        UnitConversion.insulin.Uhr_to_mUmin(basal)
                        + UnitConversion.insulin.U_to_mU(bolus) / self.dT
                    )  # mU/min
                    inputs[patient_idx, 3, sample : sample + self.dT] = insulin_rate

                    # Store IOB in slot 5 (6th input - uIOB)
                    inputs[patient_idx, 5, sample : sample + self.dT] = iob

                except Exception as e:
                    raise Exception(
                        f"Error getting action for patient {patient_idx}: {str(e)}"
                    )
        return
