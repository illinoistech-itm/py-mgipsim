from typing import Dict, Any, Optional


class Action:
    """Action namedtuple replacement for meal bolus"""

    def __init__(self, bolus: float):
        self.bolus = bolus


class MealBolusOnTheGo:
    """
    On-the-go meal bolus controller that calculates insulin bolus directly from observed carbs.

    Unlike MealAnnouncementBolusController which requires a pre-defined meal schedule,
    this controller reacts to carbs as they appear in the simulation (uFastCarb, uSlowCarb).

    This is useful when the actual meal intake timing differs from the scheduled meals,
    or when you want to calculate bolus based on real-time carb observations.
    """

    def __init__(
        self,
        carb_factor: float = 10,  # g/U (e.g., 1U per 10g CHO)
    ):
        """
        Initialize the on-the-go meal bolus controller.

        Args:
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
        """
        self.carb_factor = carb_factor

    def policy(self, carb_amount: float) -> Action:
        """
        Calculate bolus based on observed carb amount.

        Args:
            carb_amount: Current carbohydrate intake in grams

        Returns:
            Action with bolus amount in U (total units)
        """
        if carb_amount <= 0:
            return Action(bolus=0.0)

        # Calculate bolus in total units: carb amount / carb factor
        bolus_total = carb_amount / self.carb_factor  # U

        return Action(bolus=bolus_total)
