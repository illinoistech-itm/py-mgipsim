import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.Controllers.OpenAPS.Oref0 import CtrlObservation
from pymgipsim.Controllers.OpenAPS.Oref0WithMealBolusOnTheGo import (
    ORefZeroWithMealBolusOnTheGo,
)
from pymgipsim.Controllers.OpenAPS.Oref0 import ORefZeroController
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models import T1DM


class MealBolusMode(Enum):
    """Meal bolus calculation modes"""

    ON_THE_GO = 1  # Reactive: calculates bolus from observed carbs
    PLANNED = 2  # Proactive: uses pre-planned meal announcements


class Controller:
    name = "OpenAPS"

    def __init__(
        self,
        scenario_instance: scenario,
        meal_bolus_mode: MealBolusMode = MealBolusMode.ON_THE_GO,
    ):
        """
        Initialize OpenAPS controller with selectable meal bolus mode.

        Args:
            scenario_instance: Scenario configuration
            meal_bolus_mode: MealBolusMode.ON_THE_GO (reactive to observed carbs) or
                           MealBolusMode.PLANNED (pre-planned meal announcements)
        """
        if scenario_instance.patient.model.name != T1DM.ExtHovorka.Model.name:
            raise Exception("OpenAPS controller only supports the ExtHovorka model.")

        self.sampling_time = int(scenario_instance.settings.sampling_time)
        self.meal_bolus_mode = meal_bolus_mode
        self.simulation_time = datetime.now()
        self.demographic_info = scenario_instance.patient.demographic_info

        # Delegate to mode-specific initialization
        if meal_bolus_mode == MealBolusMode.ON_THE_GO:
            self._init_on_the_go_mode(scenario_instance)
        elif meal_bolus_mode == MealBolusMode.PLANNED:
            self._init_planned_mode(scenario_instance)
        else:
            raise ValueError(f"Invalid meal_bolus_mode: {meal_bolus_mode}")

    def _init_on_the_go_mode(self, scenario_instance: scenario):
        """Initialize controller for on-the-go (reactive) meal bolus mode."""
        self.controller = ORefZeroWithMealBolusOnTheGo()
        print("Using on-the-go meal bolus (reactive to observed carbs)")

        if not self.controller.health_check():
            raise ConnectionError("Failed to connect to OpenAPS server")

        model_parameters = T1DM.ExtHovorka.Parameters(
            np.asarray(scenario_instance.patient.model.parameters)
        )

        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            patient_name = self._patient_name(patient_idx)
            profile = self._create_patient_profile(patient_idx, scenario_instance)

            print(
                f"Initialized OpenAPS for {patient_name}"
                f" with basal {profile['current_basal']} U/h"
                f", isf {profile['sens']} mg/dL per U"
                f", carb ratio {profile['carb_ratio']} g/U"
                f", max basal {profile['max_basal']} U/h"
                f", max daily basal {profile['max_daily_basal']} U/day"
                f" (on-the-go meal bolus mode)"
            )
            self.controller.initialize_patient_with_carb_factor(
                patient_name=patient_name,
                profile=profile,
            )

    def _init_planned_mode(self, scenario_instance: scenario):
        """Initialize controller for planned (proactive) meal bolus mode."""
        self.controller = ORefZeroController()
        print(
            "Using planned meal bolus (proactive meal announcements, following SAPT approach)"
        )

        if not self.controller.health_check():
            raise ConnectionError("Failed to connect to OpenAPS server")

        # Load meal announcements from scenario
        time = np.arange(
            scenario_instance.settings.start_time,
            scenario_instance.settings.end_time,
            scenario_instance.settings.sampling_time,
        )
        events = scenario_instance.inputs.meal_carb
        self.meal_announcement = Signal(
            time=time,
            sampling_time=scenario_instance.settings.sampling_time,
            duration=np.ones_like(events.duration),
            start_time=events.start_time,
            magnitude=scenario_instance.settings.sampling_time
            * np.asarray(events.magnitude),
        )

        model_parameters = T1DM.ExtHovorka.Parameters(
            np.asarray(scenario_instance.patient.model.parameters)
        )

        for patient_idx in range(scenario_instance.patient.number_of_subjects):
            patient_name = self._patient_name(patient_idx)
            profile = self._create_patient_profile(patient_idx, scenario_instance)

            print(
                f"Initialized OpenAPS for {patient_name}"
                f" with basal {profile['current_basal']} U/h"
                f", isf {profile['sens']} mg/dL per U"
                f", carb ratio {profile['carb_ratio']} g/U"
                f", max basal {profile['max_basal']} U/h"
                f", max daily basal {profile['max_daily_basal']} U/day"
                f" (planned meal bolus mode - following SAPT approach)"
            )
            self.controller.initialize_patient(
                patient_name=patient_name,
                profile=profile,
            )

    def _create_patient_profile(
        self, patient_idx: int, scenario_instance: scenario
    ) -> dict:
        """Create patient profile dictionary from demographic info."""
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
            max_daily_basal = (
                self.demographic_info.total_daily_basal[patient_idx] / 24
            )  # U/h
        else:
            max_daily_basal = basal_rate

        return {
            "current_basal": basal_rate,
            "sens": isf,
            "dia": 7,
            "carb_ratio": carb_ratio,
            "max_iob": 12,
            "max_basal": max(3.5, basal_rate * 3),
            "max_daily_basal": max_daily_basal,
            "max_bg": 117,
            "min_bg": 90,
            "type": "current",
        }

    @staticmethod
    def _patient_name(index: int) -> str:
        return f"patient_{index}"

    def run(self, measurements, inputs, states, sample):
        """Run controller for current timestep - delegates to mode-specific methods."""
        if self.meal_bolus_mode == MealBolusMode.ON_THE_GO:
            self._run_on_the_go_mode(measurements, inputs, states, sample)
        elif self.meal_bolus_mode == MealBolusMode.PLANNED:
            self._run_planned_mode(measurements, inputs, states, sample)

    def _run_on_the_go_mode(self, measurements, inputs, states, sample):
        """Run controller in on-the-go mode (reactive to observed carbs)."""
        current_sim_time = self.simulation_time + timedelta(minutes=sample)

        for patient_idx in range(inputs.shape[0]):
            glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(
                measurements[patient_idx]
            )
            uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure, uIOB = inputs[
                patient_idx
            ]

            # Get current carbs at this timestep (g/min rate) from actual intake
            fast_carbs = uFastCarbs[sample]
            slow_carbs = uSlowCarbs[sample]
            sum_meals = fast_carbs + slow_carbs

            # Print carb intake when carbs are consumed
            if sum_meals > 0:
                if (
                    hasattr(self.controller, "start_time")
                    and self.controller.start_time is not None
                ):
                    elapsed_minutes = (
                        current_sim_time - self.controller.start_time
                    ).total_seconds() / 60
                    print(
                        f"[{self._patient_name(patient_idx)}] t={elapsed_minutes:.1f} min, "
                        f"Fast carbs: {fast_carbs:.2f} g/min, Slow carbs: {slow_carbs:.2f} g/min, "
                        f"Total: {sum_meals:.2f} g/min"
                    )
                else:
                    print(
                        f"[{self._patient_name(patient_idx)}] Time: {current_sim_time}, "
                        f"Fast carbs: {fast_carbs:.2f} g/min, Slow carbs: {slow_carbs:.2f} g/min, "
                        f"Total: {sum_meals:.2f} g/min"
                    )

            # Create observation for the controller
            observation = CtrlObservation(CGM=glucose, bolus=0.0)

            # Get insulin recommendation from server
            try:
                action = self.controller.policy(
                    observation=observation,
                    reward=0,
                    done=False,
                    patient_name=self._patient_name(patient_idx),
                    meal=sum_meals,
                    time=current_sim_time,
                )
                basal = action.get("basal", 0.0)  # U/min (from Oref0.py conversion)
                bolus = action.get("bolus", 0.0)  # U (total units)
                iob = action.get("iob", 0.0)  # U

                insulin_rate = (
                    UnitConversion.insulin.U_to_mU(basal)  # U/min -> mU/min
                    + UnitConversion.insulin.U_to_mU(bolus) / self.sampling_time
                )  # mU/min
                inputs[patient_idx, 3, sample] = insulin_rate
                inputs[patient_idx, 5, sample] = iob

            except Exception as e:
                raise Exception(
                    f"Error getting action for patient {patient_idx}: {str(e)}"
                )

    def _run_planned_mode(self, measurements, inputs, states, sample):
        """Run controller in planned mode (proactive meal announcements, SAPT approach)."""
        current_sim_time = self.simulation_time + timedelta(minutes=sample)

        for patient_idx in range(inputs.shape[0]):
            glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(
                measurements[patient_idx]
            )
            uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure, uIOB = inputs[
                patient_idx
            ]

            # Use meal announcement from scenario (g CHO at this sample)
            sum_meals = self.meal_announcement.sampled_signal[patient_idx, sample]

            # Print meal announcement when meals are announced
            if sum_meals > 0:
                elapsed_minutes = sample * self.sampling_time
                print(
                    f"[{self._patient_name(patient_idx)}] t={elapsed_minutes:.1f} min, "
                    f"Meal announced: {sum_meals:.2f} g CHO"
                )

            # Calculate meal bolus (following SAPT approach)
            meal_bolus = 0.0
            if sum_meals > 0:
                carb_ratio = self.demographic_info.carb_insulin_ratio[patient_idx]
                meal_bolus = sum_meals / carb_ratio  # U

            # Create observation for the controller
            # observation = CtrlObservation(CGM=glucose, bolus=meal_bolus) # send bolus to Oref0
            observation = CtrlObservation(CGM=glucose, bolus=0)  #

            # Get insulin recommendation from server
            try:
                action = self.controller.policy(
                    observation=observation,
                    reward=0,
                    done=False,
                    patient_name=self._patient_name(patient_idx),
                    meal=sum_meals,
                    time=current_sim_time,
                )
                basal = action.get("basal", 0.0)  # U/min (from Oref0.py conversion)
                bolus = action.get("bolus", 0.0)  # U (total units)

                # Add the meal bolus we calculated to ORef0's bolus
                bolus += meal_bolus

                iob = action.get("iob", 0.0)  # U

                insulin_rate = (
                    UnitConversion.insulin.U_to_mU(basal)  # U/min -> mU/min
                    + UnitConversion.insulin.U_to_mU(bolus) / self.sampling_time
                )  # mU/min
                inputs[patient_idx, 3, sample] = insulin_rate
                inputs[patient_idx, 5, sample] = iob

            except Exception as e:
                raise Exception(
                    f"Error getting action for patient {patient_idx}: {str(e)}"
                )
