import logging
from typing import Dict, Any, Optional
from pymgipsim.Controllers.OpenAPS.Oref0 import ORefZeroController, CtrlObservation
from pymgipsim.Controllers.OpenAPS.MealBolusOnTheGo import (
    MealBolusOnTheGo,
    Action,
)

logger = logging.getLogger(__name__)


class ORefZeroWithMealBolusOnTheGo:
    """
    Combined controller that uses composition to combine:
    - ORefZero for basal rate calculations
    - MealBolusOnTheGo for on-the-go meal bolus calculations (per-patient)

    This controller calculates meal bolus based on observed carb intake
    rather than pre-defined meal schedules.

    Each patient has their own MealBolusOnTheGo instance to support
    different carb factors per patient.
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
        self.meal_bolus_controllers: Dict[str, MealBolusOnTheGo] = {}

        # Track first policy call time (simulation start)
        self.start_time = None

        logger.info("ORefZeroWithMealBolusOnTheGo Controller initialized")

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
        Get insulin dosage recommendation combining ORefZero basal with on-the-go meal bolus.

        Args:
            observation: CtrlObservation object with glucose level
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            patient_name: Unique patient identifier
            meal: Carbohydrate amount in grams (from environment - sum of uFastCarb + uSlowCarb)
            time: Current simulation time (datetime object)
            print_debug: Whether to print debug information

        Returns:
            Dict with combined basal (from ORefZero) and bolus (on-the-go meal + ORefZero)
        """
        # Capture start time on first policy call
        if self.start_time is None:
            self.start_time = time

        # Get on-the-go meal bolus based on current carb intake
        # Use patient-specific meal controller if available
        if patient_name in self.meal_bolus_controllers:
            meal_bolus_action = self.meal_bolus_controllers[patient_name].policy(meal)
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

        # Calculate elapsed time in minutes
        elapsed_minutes = (time - self.start_time).total_seconds() / 60

        # Print basal and bolus when bolus is not zero
        if combined_bolus > 0:
            print(
                f"[{patient_name}] t={elapsed_minutes:.1f} min, "
                f"Meal: {meal:.2f} g, "
                f"Basal: {oref_action['basal']:.4f} U/min, "
                f"ORef0 bolus: {oref_action['bolus']:.4f} U, "
                f"Meal bolus: {meal_bolus_action.bolus:.4f} U, "
                f"Total bolus: {combined_bolus:.4f} U"
            )

        if print_debug:
            logger.debug(
                f"Combined action - ORefZero basal: {oref_action['basal']:.3f}, "
                f"ORefZero bolus: {oref_action['bolus']:.3f}, "
                f"On-the-go meal bolus: {meal_bolus_action.bolus:.3f}, "
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

    def initialize_patient_with_carb_factor(
        self,
        patient_name: str,
        profile: Dict,
    ) -> bool:
        """
        Initialize patient with both ORefZero profile and on-the-go meal bolus controller.

        Args:
            patient_name: Unique patient identifier
            profile: Patient profile for ORefZero (must contain 'carb_ratio' key)

        Returns:
            True if initialization successful, False otherwise
        """
        # Extract carb_ratio from profile (same as carb_factor)
        if "carb_ratio" not in profile:
            logger.error(
                f"Cannot initialize {patient_name}: 'carb_ratio' not found in profile"
            )
            return False

        carb_factor = profile["carb_ratio"]

        # Initialize patient on ORefZero controller
        oref_success = self.oref0_controller.initialize_patient(patient_name, profile)

        # Create patient-specific on-the-go meal bolus controller
        self.meal_bolus_controllers[patient_name] = MealBolusOnTheGo(
            carb_factor=carb_factor,
        )

        logger.info(
            f"Initialized on-the-go meal bolus controller for {patient_name} "
            f"with carb_factor={carb_factor} g/U"
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
