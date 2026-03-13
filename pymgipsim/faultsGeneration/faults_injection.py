import numpy as np
import copy
from pymgipsim.Utilities.units_conversions_constants import UnitConversion



class FaultsInjection:
    def __init__(self):
        self.faults = Faults()
        self.insulin_faults = ['max_basal', 'min_basal', 'positive_basal', 'negative_basal',
                               'unknown_stop', 'unknown_under', 'false_bolus', 'false_meal']
        self.bg_faults = ['missing_signal', 'positive_spike', 'negative_spike', 'negative_bias',
                          'positive_bias', 'min_reading', 'max_reading', 'repeated_reading', 'zero_reading', 'repeated_episode']

    def run_insulin(self, fault_type, basal=None, carb_past=None):
        if fault_type == 'max_basal':
            f_value = self.faults.insulin.maximize_basal(basal)
        elif fault_type == 'min_basal':
            f_value = self.faults.insulin.minimize_basal(basal)
        elif fault_type == 'positive_basal':
            f_value = self.faults.insulin.positive_bias_basal(basal)
        elif fault_type == 'negative_basal':
            f_value = self.faults.insulin.negative_bias_basal(basal)
        elif fault_type == 'unknown_stop':
            f_value = self.faults.insulin.unknown_stopped_delivery(basal)
        elif fault_type == 'unknown_under':
            f_value = self.faults.insulin.unknown_under_delivery(basal)
        elif fault_type == 'false_bolus':
            f_value = self.faults.insulin.false_bolus()
        elif fault_type == 'false_meal':
            f_value = self.faults.insulin.false_meal(carb_past)

        else:
            return None

        return f_value

    def run_cgm(self, fault_type, bg, bg_dist=None, bg_past=None):
        if fault_type == 'missing_signal':
            f_bg = self.faults.cgm.missing_signal(bg)
        elif fault_type == 'positive_spike':
            f_bg = self.faults.cgm.positive_spike_readings(bg)
        elif fault_type == 'negative_spike':
            f_bg = self.faults.cgm.negative_spike_readings(bg)
        elif fault_type == 'negative_bias':
            f_bg = self.faults.cgm.negative_biased_readings(bg)
        elif fault_type == 'positive_bias':
            f_bg = self.faults.cgm.positive_biased_readings(bg)
        elif fault_type == 'min_reading':
            f_bg = self.faults.cgm.minimize_readings(bg)
        elif fault_type == 'max_reading':
            f_bg = self.faults.cgm.maximize_readings(bg)
        elif fault_type == 'repeated_reading':
            f_bg = self.faults.cgm.repeated_readings(bg, bg_past)
        elif fault_type == 'zero_reading':
            f_bg = self.faults.cgm.zero_readings(bg)
        elif fault_type == 'repeated_episode':
            f_bg = self.faults.cgm.repeated_episode(bg, bg_dist, bg_past)

        else:
            return None

        return f_bg




class Fault(object):
    def __init__(self):
        self.max_basal = UnitConversion.insulin.Uhr_to_mUmin(2.0)  # Uhr
        self.min_basal = UnitConversion.insulin.Uhr_to_mUmin(0.0)
        self.bias_basal = UnitConversion.insulin.Uhr_to_mUmin(0.5)
        self.spike_bg = UnitConversion.glucose.concentration_mgdl_to_mmolL(60.0)
        self.bias_bg = UnitConversion.glucose.concentration_mgdl_to_mmolL(32.0)
        self.max_bg = UnitConversion.glucose.concentration_mgdl_to_mmolL(180.0)
        self.min_bg = UnitConversion.glucose.concentration_mgdl_to_mmolL(70.0)
        self.dynamic_bias = 0.0
        self.bias_count = 0
        self.speed = UnitConversion.glucose.concentration_mgdl_to_mmolL(0.5)
        self.bolus = 250.0


class InsulinFault(Fault):
    def maximize_basal(self, basal):
        basal[:] = self.max_basal
        return basal

    def minimize_basal(self, basal):
        basal[:] = self.min_basal
        return basal

    def positive_bias_basal(self, basal):
        basal += self.bias_basal
        return basal

    def negative_bias_basal(self, basal):
        basal = np.where(basal - self.bias_basal > 0, basal - self.bias_basal, 0.0)
        return basal

    def unknown_stopped_delivery(self, basal):
        basal[:] = self.min_basal
        return basal

    def unknown_under_delivery(self, basal):
        basal = np.where(basal - self.bias_basal > 0, basal - self.bias_basal, 0.0)
        return basal

    def false_meal(self, carb_past):
        # Repeat the last slow carb
        cur_carb = carb_past[:, -1].copy()
        # Get the number of rows and columns
        n_rows, n_cols = carb_past.shape
        # Loop through rows and update the last column
        for i in range(n_rows):
            # Get indices of non-zero elements in the row
            non_zero_indices = np.where(carb_past[i] != 0)[0]
            if non_zero_indices.size > 0:
                last_non_zero_val = carb_past[i, non_zero_indices[-1]]
                cur_carb[i] = last_non_zero_val

        return cur_carb

    def false_bolus(self):
        return self.bolus


class CGMFault(Fault):
    def missing_signal(self, bg):
        bg[:] = np.nan
        return bg

    def positive_spike_readings(self, bg):
        bg += self.spike_bg
        return bg

    def negative_spike_readings(self, bg):
        bg -= self.spike_bg
        return bg

    def negative_biased_readings(self, bg):
        # self.bias_count += 1
        # if self.dynamic_bias < self.bias_bg:
        #     self.dynamic_bias = self.speed * self.bias_count
        #     bg = np.where(bg - self.dynamic_bias > 0, bg - self.dynamic_bias, 0.0)
        # else:
        #     bg = np.where(bg - self.bias_bg > 0, bg - self.bias_bg, 0.0)
        bg = np.where(bg - self.bias_bg > 0, bg - self.bias_bg, 0.0)
        return bg

    def positive_biased_readings(self, bg):
        # self.bias_count += 1
        # if self.dynamic_bias < self.bias_bg:
        #     self.dynamic_bias = self.speed * self.bias_count
        #     bg += self.dynamic_bias
        # else:
        #     bg += self.bias_bg
        bg += self.bias_bg
        return bg

    def minimize_readings(self, bg):
        bg[:] = self.min_bg
        return bg

    def maximize_readings(self, bg):
        bg[:] = self.max_bg
        return bg

    def repeated_readings(self, bg, bg_last):
        # Make current BG equal to the last BG
        bg[:] = bg_last[:].copy()
        return bg

    def zero_readings(self, bg):
        bg[:] = 0.0
        return bg

    def repeated_episode(self, bg, bg_dist, bg_past):
        # # Get 2-hour post-meal episodes
        # n_rows, n_cols = carb_past.shape
        # episodes = np.zeros((n_rows, n_cols + copy_len))
        # # Loop through rows and update the last column
        # for i in range(n_rows):
        #     # Get indices of non-zero elements in the row
        #     non_zero_indices = np.where(carb_past[i] != 0)[0]
        #     if non_zero_indices.size > 0:
        #         episodes[i, n_cols-1:n_cols + copy_len-1] = bg[i, non_zero_indices[-1]:non_zero_indices[-1]+copy_len]
        for i in bg.shape[0]:
            bg[i] = bg_past[i, bg_past.shape[1] - bg_dist[i]]

        return bg



class Faults:

    insulin = InsulinFault()

    cgm = CGMFault()








