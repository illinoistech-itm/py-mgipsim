import numpy as np


def get_first_meal_index(carb_past):
    n_rows, n_cols = carb_past.shape
    idx = np.zeros(n_rows)
    for i in range(n_rows):
        idx[i] = np.where(carb_past[i] != 0)[0][0]
    return idx
