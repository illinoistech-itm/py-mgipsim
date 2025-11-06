import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class Action:
    """Action namedtuple replacement for meal bolus"""

    def __init__(self, bolus: float):
        self.bolus = bolus


class MealAnnouncementBolusController:
    """
    Simple meal bolus controller that releases insulin bolus before meals.

    The controller calculates bolus based on upcoming meals in the scenario,
    releasing insulin a specified time before the meal occurs.
    """

    def __init__(
        self,
        meal_schedule: Optional[List[Tuple[float, float]]] = None,
        carb_factor: float = 10,
        release_time_before_meal: float = 10,  # minutes before meal to release bolus
        carb_estimation_error: float = 0.3,  # +/- percentage of carb estimation error
        sample_time: float = 1,  # time period over which to deliver bolus (minutes)
        t_start: Optional[datetime] = None,  # patient start time (datetime)
    ):
        """
        Initialize the meal bolus controller.

        Args:
            meal_schedule: List of tuples (time_minutes, carbs_grams), e.g.,
                          [(120, 50), (360, 75), (720, 60)]
            carb_factor: Carbohydrate factor in g/U (default: 10, meaning 1U per 10g CHO)
            release_time_before_meal: Time in minutes to release bolus before meal (default: 10)
            carb_estimation_error: Percentage of error in carbohydrate estimation (e.g., 0.3 for +/- 30%)
            sample_time: Time period over which to deliver bolus in minutes (default: 1)
            t_start: Patient simulation start time as datetime object (optional)
        """
        self._meal_schedule = meal_schedule if meal_schedule is not None else []
        self.carb_factor = carb_factor
        self.release_time_before_meal = release_time_before_meal
        self.carb_estimation_error = carb_estimation_error
        self.sample_time = sample_time
        self.t_start = t_start

        # Track which meals have been delivered (by index in meal_schedule)
        self._delivered_meal_indices = set()

    def policy(self, t) -> Action:
        """
        Get bolus action for the current time.

        Delivers bolus for any undelivered meals whose bolus release time has passed.
        This ensures that even if we miss a timestep, the meal bolus still gets delivered.

        Args:
            t: Current time - can be either:
                - elapsed time in minutes (int/float), or
                - datetime object (will calculate elapsed time from t_start)

        Returns:
            Action with bolus amount in U/min (insulin rate)
        """
        # Calculate elapsed time in minutes
        if isinstance(t, datetime):
            if self.t_start is None:
                raise ValueError("t_start must be set when using datetime for policy")
            elapsed_time = (t - self.t_start).total_seconds() / 60
        else:
            elapsed_time = t

        # Check all meals in schedule for any that should be delivered
        for meal_idx, (meal_time, meal_amount) in enumerate(self._meal_schedule):
            # Skip if already delivered
            if meal_idx in self._delivered_meal_indices:
                continue

            # Calculate when bolus should be released (release_time_before_meal minutes before meal)
            bolus_release_time = meal_time - self.release_time_before_meal

            # If current time >= bolus release time, deliver the bolus
            if elapsed_time >= bolus_release_time:
                # Mark this meal as delivered
                self._delivered_meal_indices.add(meal_idx)

                # Add randomness to meal amount to simulate patient uncertainty
                adjusted_meal_amount = meal_amount
                if self.carb_estimation_error > 0:
                    random_factor = random.uniform(
                        -self.carb_estimation_error, self.carb_estimation_error
                    )
                    adjusted_meal_amount *= 1 + random_factor

                # Calculate bolus in total units: meal amount / carb factor
                bolus_total = adjusted_meal_amount / self.carb_factor  # U

                # Convert to rate (U/min) by dividing by sample_time
                bolus_rate = bolus_total / self.sample_time  # U/min

                return Action(bolus=bolus_rate)

        # No undelivered meals at this time, return zero bolus
        return Action(bolus=0)
