import numpy as np
from dataclasses import dataclass, field
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models.Inputs import BaseInputs

@dataclass
class Inputs(BaseInputs):
    """ Stores the Hovorka models input signals in Signal classes.

            Attributes:
                uFastCarbs (Signal) : Fast acting carbs [mmol/min] typically with an absorption time constant of 20 minutes.
                uSlowCarbs (Signal) : Slow acting carbs [mmol/min] typically with an absorption time constant of 40 minutes.
                uHR (Signal) : Heart rate signal [BPM], used in the physical activity submodel.
                uInsulin (Signal) : Combined basal and bolus insulin input [mU/min]
                energy_expenditure (Signal) : Energy expenditure signal (currently unused in model)
                uIOB (Signal) : Insulin On Board tracking [U] - populated by OpenAPS controller

            Hint:
                .as_array() function returns 3D numpy array where:
                1st dim: Subject in the virtual cohort
                2nd dim: Input variable (6 inputs total)
                3rd dim: Timestep in the simulation horizon

    """
    uFastCarbs: Signal = field(default_factory=lambda: Signal())
    uSlowCarbs: Signal = field(default_factory=lambda: Signal())
    uHR: Signal = field(default_factory=lambda: Signal())
    uInsulin: Signal = field(default_factory=lambda: Signal())
    energy_expenditure: Signal = field(default_factory=lambda: Signal())
    uIOB: Signal = field(default_factory=lambda: Signal())



    @property
    def as_array(self):
        self._as_array = np.stack((self.uFastCarbs.sampled_signal, self.uSlowCarbs.sampled_signal, self.uHR.sampled_signal, self.uInsulin.sampled_signal, self.energy_expenditure.sampled_signal, self.uIOB.sampled_signal),axis=1)
        return self._as_array

    @as_array.setter
    def as_array(self, array: np.ndarray):
        array_sw = np.swapaxes(array, 0, 1)
        self.uFastCarbs.sampled_signal, self.uSlowCarbs.sampled_signal, self.uHR.sampled_signal, self.uInsulin.sampled_signal, self.energy_expenditure.sampled_signal, self.uIOB.sampled_signal = array_sw
        self._as_array = array
