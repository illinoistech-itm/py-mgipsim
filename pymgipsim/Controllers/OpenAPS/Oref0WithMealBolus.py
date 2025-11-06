import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pymgipsim.Controllers.OpenAPS.Oref0 import ORefZeroController, CtrlObservation
from pymgipsim.Controllers.OpenAPS.MealBolus import MealAnnouncementBolusController, Action

logger = logging.getLogger(__name__)


class ORefZeroWithMealBolus:
    """
    Combined controller that uses composition to combine:
    - ORefZero for basal rate calculations
    - MealAnnouncementBolusController for predictive meal bolus calculations (per-patient)

    This controller uses composition (has-a relationship) rather than inheritance
    to keep the two controllers separate and avoid variable conflicts.

    Each patient has their own MealAnnouncementBolusController instance to support
    different meal schedules and parameters per patient.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        timeout: int = 30,
        default_profile: Optional[Dict] = None,
    ):
        """
        Initialize the combined controller.

        Args:
            server_url: URL of the Node.js OpenAPS server
            timeout: Request timeout in seconds
            default_profile: Default patient profile for ORefZero
        """
        # Create ORefZeroController instance
        self.oref0_controller = ORefZeroController(
            server_url=server_url, timeout=timeout, default_profile=default_profile
        )

        # Dictionary to store per-patient meal bolus controllers
        self.meal_bolus_controllers: Dict[str, MealAnnouncementBolusController] = {}

        logger.info("ORefZeroWithMealBolus Controller initialized")

    def policy(
        self,
        observation,
        reward: float,
        done: bool,
        patient_name: str,
        meal: float,
        time,
        print_debug: bool = False,
    ) -> Dict[str, float]:
        """
        Get insulin dosage recommendation combining ORefZero basal with meal bolus.

        Args:
            observation: CtrlObservation object with glucose level
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            patient_name: Unique patient identifier
            meal: Carbohydrate amount in grams (from environment)
            time: Current simulation time (datetime object)
            print_debug: Whether to print debug information

        Returns:
            Dict with combined basal (from ORefZero) and bolus (meal announcement + ORefZero)
        """

        # Get meal announcement bolus (if any meal is upcoming)
        # Use patient-specific meal controller if available
        if patient_name in self.meal_bolus_controllers:
            meal_bolus_action = self.meal_bolus_controllers[patient_name].policy(time)
        else:
            # No meal controller for this patient, use zero bolus
            meal_bolus_action = Action(bolus=0.0)

        # Create new observation with meal bolus (namedtuples are immutable)
        observation_with_bolus = CtrlObservation(
            CGM=observation.CGM, bolus=meal_bolus_action.bolus
        )

        # Get ORefZero recommendation (basal and any bolus from ORefZero)
        # ORefZero expects datetime object
        oref_action = self.oref0_controller.policy(
            observation_with_bolus, reward, done, patient_name, meal, time, print_debug
        )

        # Combine: use ORefZero basal, add meal bolus to ORefZero bolus
        combined_bolus = oref_action["bolus"] + meal_bolus_action.bolus

        if print_debug:
            logger.debug(
                f"Combined action - ORefZero basal: {oref_action['basal']:.3f}, "
                f"ORefZero bolus: {oref_action['bolus']:.3f}, "
                f"Meal bolus: {meal_bolus_action.bolus:.3f}, "
                f"Total bolus: {combined_bolus:.3f}"
            )

        return {
            "basal": oref_action["basal"],
            "bolus": combined_bolus,
            "iob": oref_action["iob"],
        }

    def initialize_patient(
        self, patient_name: str, profile: Optional[Dict] = None
    ) -> bool:
        """Initialize patient on ORefZero controller."""
        return self.oref0_controller.initialize_patient(patient_name, profile)

    def initialize_patient_with_meal_schedule(
        self,
        patient_name: str,
        profile: Optional[Dict],
        meal_schedule: List[Tuple[float, float]],
        carb_factor: float,
        release_time_before_meal: float = 10,
        carb_estimation_error: float = 0.3,
        sample_time: float = 1,
        t_start: Optional[datetime] = None,
    ) -> bool:
        """
        Initialize patient with both ORefZero profile and meal schedule controller.

        Args:
            patient_name: Unique patient identifier
            profile: Patient profile for ORefZero
            meal_schedule: List of tuples (time_minutes, carbs_grams)
            carb_factor: Carbohydrate factor in g/U (from demographic_info.carb_insulin_ratio)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Percentage of error in carbohydrate estimation (e.g., 0.3 for +/- 30%)
            sample_time: Time period over which to deliver bolus in minutes (default: 1)
            t_start: Patient simulation start time as datetime object

        Returns:
            True if initialization successful, False otherwise
        """
        # Initialize patient on ORefZero controller
        oref_success = self.oref0_controller.initialize_patient(patient_name, profile)

        # Create patient-specific meal bolus controller
        self.meal_bolus_controllers[patient_name] = MealAnnouncementBolusController(
            meal_schedule=meal_schedule,
            carb_factor=carb_factor,
            release_time_before_meal=release_time_before_meal,
            carb_estimation_error=carb_estimation_error,
            sample_time=sample_time,
            t_start=t_start,
        )

        logger.info(
            f"Initialized meal controller for {patient_name} with {len(meal_schedule)} meals, "
            f"carb_factor={carb_factor} g/U"
        )

        return oref_success

    def health_check(self) -> bool:
        """Check if the OpenAPS server is responding."""
        return self.oref0_controller.health_check()

    def get_patient_status(self, patient_name: str) -> Optional[Dict[str, Any]]:
        """Get current patient status from ORefZero controller."""
        return self.oref0_controller.get_patient_status(patient_name)

    def update_patient_profile(
        self, patient_name: str, profile_updates: Dict[str, Any]
    ) -> bool:
        """Update patient profile on ORefZero controller."""
        return self.oref0_controller.update_patient_profile(
            patient_name, profile_updates
        )

    @property
    def target_bg(self) -> float:
        """Get the target blood glucose level from ORefZero controller."""
        # Note: This returns the default profile target, not patient-specific
        return (
            self.oref0_controller.default_profile["min_bg"]
            + self.oref0_controller.default_profile["max_bg"]
        ) / 2
