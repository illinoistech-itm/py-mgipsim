from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from .States import States
from .Parameters import Parameters
from .Inputs import Inputs
from pymgipsim.Utilities.Timestamp import Timestamp
from pymgipsim.VirtualPatient.Models.Model import BaseModel
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.InputGeneration.signal import Signal
from numba import njit
import numpy as np
from numba import njit

from .CONSTANTS import *

class Model(BaseModel, UnitConversion):
    """ Implements extended Hovorka model.
            Submodels: Carbohydrate absorption, Insulin Absorption...

            Note:
                20 patients available in...

            Attributes:
                inputs (Inputs): Container for model inputs.
                parameters (Parameters): Container for model parameter values.
                states (States): Container for model state values.
                initial_conditions (States): Container for initial conditions of the state variables.
                time (Timestamp): Timestamp object that stores the timesteps of the simulation horizon.
                sampling_time (int): Sampling time of the simulation.

            Examples:
                >>> model.parameters.as_array()
                ---------------------------
                Returns a 2D numpy array where:
                1st dim: Subjects
                2nd dim: Parameters
                ---------------------------
    """

    name = "T1DM.ExtHovorka"
    glucose_state = 6
    output_state = 8
    def __init__(self, sampling_time):
        self.inputs = Inputs()
        self.parameters = Parameters()
        self.states = States()
        self.initial_conditions = States()
        self.time = Timestamp()
        self.sampling_time = sampling_time

        self.EGP_insulin_effect: bool = True
        self.RENAL_effect: bool = True
        self.F01_effect: bool = True
        self.Physact_effect: bool = True

        self.faults_label = None

    @staticmethod
    @njit("float64[:,:](float64[:,:],float64,float64[:,:],float64[:,:])", cache=True)
    def model(states, time, parameters, inputs):

        """ Define Parameters """
        (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
        VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

        # Hard lower and upper constraint on Q1
        states[states[:, 6] < VG, 6] = VG[states[:, 6] < VG]
        states[states[:, 6] > VG * 32, 6] = VG[states[:, 6] > VG * 32] * 32

        """ Define State variables """
        S1, S2, I, x1, x2, x3, Q1, Q2, IG, D1Slow, D2Slow, D1Fast, D2Fast, EEfast, EEhighintensity, EElongeffect = states.T

        # Q1 = np.max(np.row_stack((Q1, np.ones_like(Q1) * VG)), axis=0)
        # Q1 = np.min(np.row_stack((Q1, np.ones_like(Q1) * VG * 32)), axis=0)

        # Nonlinear insulin sensitivity mapping
        SIModelRatioHypo = (aSI * np.tanh(Q1 / VG / b + dSI) + c) / (aSI * np.tanh(5 / b + dSI) + c)
        SIModelRatioHyper = 1 - 0.018 * (Q1 / VG - 5.55)

        """ Overwriting parameters"""
        kb1_nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
        kb2_nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
        kb3_nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper


        """ Define Inputs """
        uFastCarbs, uSlowCarbs, uHR, uInsulin, _ = inputs.T

        """ Define Equations """
        ExIntensityReserve = uHR - HRrest
        ExIntensityReserve = ExIntensityReserve/(HRmax-HRrest)
        ExIntensityReserve[ExIntensityReserve < 0] = 0
        ExIntensityReserve[ExIntensityReserve > 1] = 1
        ExIntensityReserve_High = (ExIntensityReserve - VT_HRR) / (1 - VT_HRR)
        ExIntensityReserve_High[ExIntensityReserve_High < 0] = 0
        ExIntensityReserve_High[ExIntensityReserve_High > 1] = 1

        
        # Insulin System
        dS1dt = uInsulin - S1 / tmaxI
        dS2dt = (S1 - S2) / tmaxI
        dIdt = S2 / (VI * tmaxI) - ke * I

        # Insulin Action System
        dx1dt = I * kb1_nonlin - x1 * ka1
        dx2dt = I * kb2_nonlin - x2 * ka2
        dx3dt = I * kb3_nonlin - x3 * ka3

        # Glucose compartment
        # Nervous system glucose uptake
        F01_nonlin = BW * F01
        hypo_binmap = (Q1 / VG) < NOMINAL_F01_threshold
        F01_nonlin[hypo_binmap] = F01_nonlin[hypo_binmap] * (Q1[hypo_binmap] / VG[hypo_binmap]) / NOMINAL_F01_threshold

        # Renal glucose clearance
        FR = np.zeros_like(Q1)
        FR_binmap = (Q1 / VG) >= NOMINAL_FR_threshold
        FR[FR_binmap] = 0.003 * ((Q1[FR_binmap] / VG[FR_binmap]) - NOMINAL_FR_threshold) * VG[FR_binmap]

        # Gut glucose absorption
        UGe = (D2Slow / tmaxG + D2Fast / tmaxGFast)

        # Endogenuous glucose production meditated by insulin and physical activity
        EGP_contribution = EGP0 * (1 - x3) * (1 + p1 * EEfast + p2 * EEhighintensity)
        EGP_contribution[EGP_contribution < 0] = 0

        # Glucose compartments
        dQ1dt = (-Q1 * x1 + k12 * Q2 - F01_nonlin - FR + UGe + EGP_contribution - p3 * EEfast)
        dQ2dt = (Q1 * x1 - k12 * Q2 - x2 * Q2 + p3 * EEfast - p4 * EEfast - p5 * x2 * EElongeffect)

        dIGdt = ((1 / tsub) * ((Q1 / VG) - IG))

        # Slow carbohydrate absorption system
        dD1Slowdt = (AG * (uSlowCarbs) - (D1Slow / tmaxG))
        dD2Slowdt = ((D1Slow - D2Slow) / tmaxG)

        # Fast carbohydrate absorption system
        dD1Fastdt = (AG * (uFastCarbs) - (D1Fast / tmaxGFast))
        dD2Fastdt = ((D1Fast - D2Fast) / tmaxGFast)

        # Physical activity system
        dEEfastdt = p6 * ExIntensityReserve - p6 * EEfast
        dEEhighintensitydt = p8 * ExIntensityReserve_High - p8 * EEhighintensity
        dEElongeffectdt = p6 * EEfast - p7 * EElongeffect

        return np.column_stack((dS1dt, dS2dt, dIdt, dx1dt, dx2dt, dx3dt, dQ1dt, dQ2dt, dIGdt, dD1Slowdt, dD2Slowdt,
                                dD1Fastdt, dD2Fastdt, dEEfastdt, dEEhighintensitydt, dEElongeffectdt))
    
    @staticmethod
    def rate_equations(states, time, parameters, inputs):
        pass

    @staticmethod
    def get_basal_equilibrium(parameters, basal_blucose):
        """ Define Parameters """
        (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
        VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = parameters.T

        F01_nonlin = BW * F01

        G = basal_blucose

        SIModelRatioHypo = (aSI * np.tanh(G / b + dSI) + c) / (aSI * np.tanh(5 / b + dSI) + c)
        SIModelRatioHyper = 1 - 0.018 * (G - 5.55)

        """ Overwriting parameters"""
        kb1nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
        kb2nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
        kb3nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper

        basal_equilibrium = (VI*ka1**(1/2)*ke*(ka1*EGP0**2*k12**2*ka2**2*kb3nonlin**2
                                               + 2*ka1*EGP0**2*k12*ka2*ka3*kb2nonlin*kb3nonlin
                                               + ka1*EGP0**2*ka3**2*kb2nonlin**2
                                               - 2*ka1*EGP0*F01_nonlin*k12*ka2*ka3*kb2nonlin*kb3nonlin
                                               - 2*ka1*EGP0*F01_nonlin*ka3**2*kb2nonlin**2
                                               + 4*G*VG*kb1nonlin*EGP0*k12*ka2*ka3**2*kb2nonlin
                                               + ka1*F01_nonlin**2*ka3**2*kb2nonlin**2
                                               - 4*G*VG*kb1nonlin*F01_nonlin*k12*ka2*ka3**2*kb2nonlin)**(1/2)
                             + EGP0*VI*ka1*ka3*kb2nonlin*ke - F01_nonlin*VI*ka1*ka3*kb2nonlin*ke
                             - EGP0*VI*k12*ka1*ka2*kb3nonlin*ke)/(2*(EGP0*ka1*kb2nonlin*kb3nonlin + G*VG*ka3*kb1nonlin*kb2nonlin))

        return basal_equilibrium

    def preprocessing(self):

        (kb1, kb2, kb3, EGP0, ke, F01, AG, tmaxI, tmaxG, p3, p4, p5, beta, a, BW, HRrest, HRmax, VG, VI,
        VT_HRR, k12, ka1, ka2, ka3, aSI, b, c, dSI, tsub, p1, p2, p6, p7, p8, tmaxGFast) = self.parameters.as_array.T

        S10 = self.inputs.uInsulin.sampled_signal[:,0]*self.parameters.tmaxI
        S20 = S10
        I0 = S20/self.parameters.tmaxI/self.parameters.ke/self.parameters.VI

        G0 = 6*np.ones_like(self.parameters.VG)

        Q10 = G0*self.parameters.VG

        SIModelRatioHypo = (aSI * np.tanh(G0 / b + dSI) + c) / (aSI * np.tanh(5 / b + dSI) + c)
        SIModelRatioHyper = 1 - 0.018 * (G0 - 5.55)

        """ Overwriting parameters"""
        kb1nonlin = kb1 * SIModelRatioHypo * SIModelRatioHyper
        kb2nonlin = kb2 * SIModelRatioHypo * SIModelRatioHyper
        kb3nonlin = kb3 * SIModelRatioHypo * SIModelRatioHyper

        x10 = kb1nonlin * I0 / self.parameters.ka1
        x20 = kb2nonlin * I0 / self.parameters.ka2
        x30 = kb3nonlin * I0 / self.parameters.ka3


        Q20 = x10*Q10/(self.parameters.k12+x20)

        D1Slow0 = np.zeros_like(G0)
        D2Slow0 = np.zeros_like(G0)

        D1Fast0 = np.zeros_like(G0)
        D2Fast0 = np.zeros_like(G0)

        self.initial_conditions.as_array = np.column_stack((S10,S20,I0,x10,x20,x30,Q10,Q20,G0,D1Slow0,D2Slow0,D1Fast0,D2Fast0,np.zeros_like(G0),np.zeros_like(G0),np.zeros_like(G0)))
        self.states.as_array = np.zeros((self.inputs.as_array.shape[0], self.initial_conditions.as_array.shape[1], self.inputs.as_array.shape[2]))





    @staticmethod
    def output_equilibrium(parameters, inputs):
        pass

    def update_scenario(self, scenario):
        pass

    @staticmethod
    def from_scenario(scenario_instance: scenario):
        time = np.arange(scenario_instance.settings.start_time,
                         scenario_instance.settings.end_time,
                         scenario_instance.settings.sampling_time
                         )

        model = Model(sampling_time=scenario_instance.settings.sampling_time)

        events = scenario_instance.inputs.meal_carb
        meal_carb = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                           start_time=events.start_time, magnitude=UnitConversion.glucose.g_glucose_to_mmol(np.asarray(events.magnitude)))

        events = scenario_instance.inputs.snack_carb
        snack_carb = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                            start_time=events.start_time, magnitude=UnitConversion.glucose.g_glucose_to_mmol(np.asarray(events.magnitude)))

        events = scenario_instance.inputs.heart_rate
        heart_rate = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                        start_time=events.start_time, magnitude=events.magnitude)

        events = scenario_instance.inputs.basal_insulin
        converted = UnitConversion.insulin.Uhr_to_mUmin(np.asarray(events.magnitude))
        basal = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                        start_time=events.start_time, magnitude=converted)
        events = scenario_instance.inputs.bolus_insulin
        converted = UnitConversion.insulin.U_to_mU(np.asarray(events.magnitude))
        bolus = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time, duration=events.duration,
                        start_time=events.start_time, magnitude=converted)

        insulin = Signal()
        insulin.sampled_signal = basal.sampled_signal + bolus.sampled_signal

        events = scenario_instance.inputs.energy_expenditure
        energy_expenditure = Signal(time=time, sampling_time=scenario_instance.settings.sampling_time,start_time=events.start_time, magnitude=events.magnitude)

        model.inputs = Inputs(snack_carb, meal_carb, heart_rate, insulin, energy_expenditure)
        model.parameters = Parameters(np.asarray(scenario_instance.patient.model.parameters))
        model.time = Timestamp()
        model.time.as_unix = time

        return model
