import zoneinfo
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class ORefZeroController:
    """
    OpenAPS oref0 controller that communicates with Node.js server
    for insulin dosage recommendations
    """

    DEFAULT_TIMEZONE = "UTC"
    MINIMAL_TIMESTEP = 5  # in minutes
    BASE_PROFILE = {
        "current_basal": 0.7,  # Current basal rate in U/h
        "sens": 50,  # Insulin Sensitivity Factor (ISF)
        "dia": 6,  # Duration of Insulin Action in hours
        "carb_ratio": 10,  # Carb Ratio (g/U)
        "max_iob": 6,  # Maximum insulin on board allowed
        "max_basal": 3.5,  # Maximum temporary basal rate in U/h
        "max_daily_basal": 3.5,  # Maximum daily basal rate in units per day
        "max_bg": 120,  # Upper target
        "min_bg": 120,  # Lower target
        "maxCOB": 120,  # Maximum carbs on board
        "isfProfile": {"sensitivities": [{"offset": 0, "sensitivity": 50}]},
        "min_5m_carbimpact": 12.0,  # Minimum carb absorption rate
        "type": "current",  # Profile type
    }

    def __init__(
        self,
        server_url: str = "http://localhost:3000",
        timeout: int = 30,
        default_profile: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ORefZero controller

        Args:
            current_basal: current basal rate
            server_url: URL of the Node.js OpenAPS server
            profile: Custom patient profile, uses default if None
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        self.collect_meal = 0

        # Patient state tracking
        self.patient_profiles = {}  # patientId -> profile mapping
        self.last_glucose_time = {}  # patientId -> last glucose timestamp
        self.glucose_history = {}  # patientId -> list of glucose readings
        self.meal_history = {}  # patientId -> list of meal entries
        self.pump_history = {}  # patientId -> list of pump events

        if default_profile:
            self.BASE_PROFILE.update(default_profile)
        self.default_profile = self.BASE_PROFILE

    @property
    def target_bg(self, patient_name) -> float:
        """Get the target blood glucose level."""
        if patient_name not in self.patient_profiles:
            raise ValueError("Patient not initialized")
        return (
            self.patient_profiles.get([patient_name])["min_bg"]
            + self.patient_profiles.get([patient_name])["max_bg"]
        ) / 2

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to the server with error handling"""
        url = f"{self.server_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.timeout)
            elif method.upper() == "PATCH":
                response = self.session.patch(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise Exception("Server request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to OpenAPS server")
        except requests.exceptions.HTTPError as e:
            pass

        except Exception as e:
            raise

    def initialize_patient(
        self, patient_name: str, profile: Optional[Dict] = None
    ) -> bool:
        """Initialize patient on the server"""
        if patient_name in self.patient_profiles:
            return True

        patient_profile = self.default_profile.copy()
        patient_profile.update(profile)

        # Prepare initialization data
        init_data = {
            "profile": patient_profile,
            "initialData": {"glucoseHistory": [], "pumpHistory": [], "carbHistory": []},
            "settings": {
                "timezone": self.DEFAULT_TIMEZONE,
                "historyRetentionHours": 24,
                "autoCleanup": True,
            },
        }

        try:
            response = self._make_request(
                "POST", f"/patients/{patient_name}/initialize", init_data
            )
            self.patient_profiles[patient_name] = patient_profile
            self.glucose_history[patient_name] = []
            self.meal_history[patient_name] = []
            self.pump_history[patient_name] = []
            self.last_glucose_time[patient_name] = None
            return True

        except Exception as e:
            return False

    @staticmethod
    def _local_timezone() -> datetime:
        return datetime.now().astimezone()

    def _convert_to_utc(self, local_time: datetime) -> datetime:
        if local_time.tzinfo is None:
            # If naive datetime, assume local timezone
            local_time = local_time.replace(tzinfo=self._local_timezone().tzinfo)
        utc_time = local_time.astimezone(zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE))
        # Remove the timezone info (make it naive)
        return utc_time.replace(tzinfo=None)

    def _convert_time_to_timestamp(self, time_obj) -> str:
        """Convert time object to ISO timestamp string"""
        if isinstance(time_obj, str):

            # Assume it's already a valid timestamp
            try:
                time_obj = datetime.fromisoformat(time_obj.rstrip("Z"))
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )
        elif isinstance(time_obj, (int, float)):
            # Assume it's a Unix timestamp
            try:
                time_obj = datetime.fromtimestamp(time_obj)
            except:
                raise ValueError(
                    f"Invalid timestamp format, could not convert {time_obj} to datetime"
                )

        # now we have a datetime object
        if not isinstance(time_obj, datetime):
            raise ValueError(f"Expected datetime object, got {type(time_obj)}")
        # Convert to UTC and format as ISO string
        utc_time = self._convert_to_utc(time_obj)
        return utc_time.isoformat() + "Z"  # Append 'Z' to indicate

    def _prepare_new_data(
        self, patient_name: str, glucose: float, meal: float, timestamp: str
    ) -> Dict[str, Any]:
        """Prepare new data to send to the server"""
        new_data = {}

        # Add glucose reading if we have one
        if glucose > 0:
            # Convert UTC timestamp string to local datetime, then to int (epoch seconds)
            utc_dt = datetime.fromisoformat(timestamp.rstrip("Z")).replace(
                tzinfo=zoneinfo.ZoneInfo(self.DEFAULT_TIMEZONE)
            )
            glucose_entry = {
                "date": int(utc_dt.timestamp() * 1000),  # Convert to milliseconds
                "glucose": glucose,
                "timestamp": timestamp,
            }
            new_data["glucoseReadings"] = [glucose_entry]
            self.glucose_history[patient_name].append(glucose_entry)

        # Add carb entry if we have a meal
        if meal > 0:
            carb_entry = {
                "created_at": timestamp,
                "carbs": meal,
                "enteredBy": "controller",
            }
            new_data["carbEntries"] = [carb_entry]
            self.meal_history[patient_name].append(carb_entry)

        return new_data

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
        Get insulin dosage recommendation from OpenAPS server

        Args:
            observation: CtrlObservation object with glucose level
            reward: Reward signal (not used by OpenAPS)
            done: Episode done flag (not used by OpenAPS)
            patient_name: Unique patient identifier
            meal: Carbohydrate amount in grams
            time: Current simulation time

        Returns:
            Dict with basal and bolus insulin recommendations
        """
        # Initialize patient if not already done
        if not self.initialize_patient(patient_name):
            # return Action(basal=1.0, bolus=0.0)  # Default safe action
            raise ValueError("Forgot to initialise patient.")

        # Convert time to timestamp
        timestamp = self._convert_time_to_timestamp(time)
        previous_timestamp = self.last_glucose_time.get(patient_name)

        # accumulate meal data
        self.collect_meal += meal

        # if previous_timestamp is not None and timestamp difference < MINIMAL_TIMESTEP
        if previous_timestamp is not None and (
            datetime.fromisoformat(timestamp.rstrip("Z"))
            < datetime.fromisoformat(previous_timestamp.rstrip("Z"))
            + timedelta(minutes=self.MINIMAL_TIMESTEP)
        ):
            # TODO: check with loopinsight which one is correct
            return {
                "basal": self.last_insulin["basal"],
                "bolus": self.last_insulin["bolus"],
            }
            # return {"basal": 0, "bolus": 0}

        # Extract glucose level
        glucose_level = observation.CGM  # if hasattr(observation, "CGM") else 100.0

        # Prepare new data
        new_data = self._prepare_new_data(
            patient_name, glucose_level, self.collect_meal, timestamp
        )
        self.collect_meal = 0  # Reset after sending
        if print_debug:
            print(new_data)
        # Prepare calculation request
        calc_data = {
            "currentTime": timestamp,
            "newData": new_data,
            "options": {
                "microbolus": False,  # Disable, since oref0 doesn't support microbolus, though it might be available in the future for oref1
                "overrideProfile": {},  # Could be used for temporary profile changes
            },
        }

        # Make calculation request
        endpoint = f"/patients/{patient_name}/calculate"
        response = self._make_request("POST", endpoint, calc_data)
        if print_debug:
            print(response["suggestion"])
        # Extract recommendation
        suggestion = response.get("suggestion", {})
        basal_rate = response.get("IIR", 0.0) / 60  # U/h -> U/min

        # Calculate bolus recommendation
        # OpenAPS typically provides basal adjustments, bolus calculation
        # might need to be derived from the suggestion
        bolus_amount = 0.0

        # Check if there's a microbolus recommendation
        if "units" in suggestion:
            bolus_amount = suggestion.get("units", 0.0)

        # Check for SMB (Super Micro Bolus) recommendation
        if "microbolus" in suggestion:
            bolus_amount += suggestion.get("microbolus", 0.0)

        # Store the last glucose time
        self.last_glucose_time[patient_name] = timestamp
        self.last_insulin = {"basal": basal_rate, "bolus": bolus_amount}

        return {"basal": basal_rate, "bolus": bolus_amount}  # U/min, U

    def get_patient_status(self, patient_name: str) -> Optional[Dict[str, Any]]:
        """Get current patient status from the server"""
        try:
            if patient_name not in self.patient_profiles:
                return None

            response = self._make_request("GET", f"/patients/{patient_name}/status")
            return response

        except Exception as e:
            return None

    def update_patient_profile(
        self, patient_name: str, profile_updates: Dict[str, Any]
    ) -> bool:
        """Update patient profile on the server"""
        try:
            if patient_name not in self.patient_profiles:
                return False

            response = self._make_request(
                "PATCH", f"/patients/{patient_name}/profile", profile_updates
            )

            # Update local copy
            self.patient_profiles[patient_name].update(profile_updates)
            return True

        except Exception as e:
            return False

    def health_check(self) -> bool:
        """Check if the OpenAPS server is responding"""
        try:
            response = self._make_request("GET", "/health")
            return True
        except Exception as e:
            return False
