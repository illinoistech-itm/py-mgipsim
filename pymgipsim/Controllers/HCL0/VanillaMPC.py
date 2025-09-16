# Linear MPC, implemented by Mohammad Ahmadasas
import numpy as np
from pymgipsim.Controllers.HCL0.DataContainer import *
import qpsolvers
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.Utilities.Scenario import scenario
from pymgipsim.InputGeneration.signal import Signal
from pymgipsim.VirtualPatient.Models import T1DM


class VanillaMPC:
    # Linear MPC, implemented by Mohammad Ahmadasas

    def __init__(self, scenario_instance: scenario, patient_idx):
        self.current_states_object = Hovorka_Model_Extended_States_Object_MealComp0()
        demographic_info = scenario_instance.patient.demographic_info
        self.demographic_info = Demographic_Information(BW=demographic_info.body_weight[patient_idx], height=demographic_info.height[patient_idx], insulin2carb_ratio=demographic_info.carb_insulin_ratio[patient_idx],
                                                        correction_factor=demographic_info.correction_bolus[patient_idx], basal_rate=demographic_info.basal[patient_idx],
                                                        HbA1c=demographic_info.HbA1c[patient_idx], waist_size=demographic_info.waist_size[patient_idx])
        model_parameters = T1DM.ExtHovorka.Parameters(np.asarray(scenario_instance.patient.model.parameters))
        self.hovorka_params = Hovorka_Parameters(TmaxI=model_parameters.tmaxI[patient_idx],
                                                 SIT=model_parameters.kb1[patient_idx]/model_parameters.ka1[patient_idx],
                                                 SID=model_parameters.kb2[patient_idx]/model_parameters.ka2[patient_idx],
                                                 SIE=model_parameters.kb3[patient_idx]/model_parameters.ka3[patient_idx],
                                                 Ke=model_parameters.ke[patient_idx], K12=model_parameters.k12[patient_idx],
                                                 Ag=0.8, TmaxG=model_parameters.tmaxG[patient_idx], gain=1,
                                                 TmaxE=5, EGP0=model_parameters.EGP0[patient_idx]/model_parameters.BW[patient_idx],
                                                 F01=model_parameters.F01[patient_idx], BW=model_parameters.BW[patient_idx])

        self.pw_data_object = Data_PW()
        self.bounds = Bounds()
        self.ysp_mg_dl = 110  # mg/dl
        self.max_bolus = 0.5
        self.T = 5
        self.prediction_horizon = 24

        self.linear_model = Linear_State_Space_Model(T=self.T)
        self.mpc_obj_fcn_params = MPC_Objective_Function_Params(prediction_horizon=self.prediction_horizon, Q_e=1)


    def mpc_execute(self):
        n = self.linear_model.get_state_num()
        m = self.linear_model.get_input_num_artificial()
        p = self.linear_model.get_output_num()
        T = self.linear_model.T

        current_cgm = self.pw_data_object.get_last_cgm()
        ysp_deviation_mmol_l = (self.ysp_mg_dl - current_cgm) / 18.0  # mmol/l
        self.mpc_obj_fcn_params.ysp = ysp_deviation_mmol_l

        # setting bounds
        self.bounds.x_min = np.full(n, -np.inf)
        self.bounds.x_max = np.full(n, np.inf)
        self.bounds.y_min = np.full(p, -np.inf)
        self.bounds.y_max = np.full(p, np.inf)

        self.bounds.u_min = np.zeros((m, self.prediction_horizon))
        self.bounds.u_max = np.zeros((m, self.prediction_horizon))

        if self.pw_data_object.get_dcgm_dt() < 0 and self.pw_data_object.get_cgm_min(6) < 180.0:
            self.bounds.u_min[0, :] = (0 - self.pw_data_object.get_last_insulin_input_mU_min())
            self.bounds.u_max[0, :] = (self.max_bolus * 1000.0 / T - self.pw_data_object.get_last_insulin_input_mU_min())
        else:
            self.bounds.u_min[0, :] = (
                        self.pw_data_object.get_last_basal_mU_min() - self.pw_data_object.get_last_insulin_input_mU_min())
            self.bounds.u_max[0, :] = (self.max_bolus * 1000.0 / T + self.demographic_info.get_basal_rate_mU_min() -
                                       self.pw_data_object.get_last_insulin_input_mU_min())

        self.bounds.u_min[1, :] = self.bounds.u_max[1, :] = 0 - self.pw_data_object.meal_pw[-2]
        self.bounds.u_min[2, :] = self.bounds.u_max[2, :] = 1 - self.pw_data_object.energy_expenditure_pw[-2]
        self.bounds.u_min[3, :] = self.bounds.u_max[3, :] = 1

        # Hypoglycemia safety: restrict insulin if glucose is low or dropping
        # if current_cgm < 100 or self.pw_data_object.get_dcgm_dt() < 0:
        #     self.bounds.u_max[0, :] = self.demographic_info.get_basal_rate_mU_min()

        qop = self.mpc_set()
        x_solution, exit_flag = self.mpc_solve(qop)

        # check if the qpsolver exit flag is successful, if not continue with the last calculated basal
        if x_solution is not None:
            sample_time_decision_var_num = p + p + n + m
            u_opt = x_solution.reshape((self.prediction_horizon, sample_time_decision_var_num))[:, -4:-1].T
            u_horizon = u_opt + np.array([
                self.pw_data_object.get_last_insulin_input_mU_min(),
                self.pw_data_object.meal_pw[-2],
                self.pw_data_object.energy_expenditure_pw[-2]
            ]).reshape(-1, 1)
            opt_insulin_trajectory = u_horizon[0, :]
        else:
            u_horizon = np.concatenate([
                [[self.pw_data_object.get_last_basal_mU_min()] * self.prediction_horizon],
                [[self.pw_data_object.get_last_meal()] * self.prediction_horizon],
                [[self.pw_data_object.get_last_energy_expenditure()] * self.prediction_horizon]
            ], axis=0)
            opt_insulin_trajectory = u_horizon[0, :]

        opt_insulin = opt_insulin_trajectory[0]

        # basal_out unit is U/hr and bolus_out unit is U
        if opt_insulin <= 0:
            basal_out = 0
            bolus_out = 0
        elif (opt_insulin > 0) and (opt_insulin <= self.demographic_info.get_basal_rate_mU_min()):
            basal_out = opt_insulin * 60.0 / 1000
            bolus_out = 0
        else:
            basal_out = self.demographic_info.get_basal_rate()  # U/hr
            # safety bound: set cap bolus
            bolus_out = (opt_insulin - self.demographic_info.get_basal_rate_mU_min()) / 1000 * T
            # extra = opt_insulin - self.demographic_info.get_basal_rate_mU_min()
            # bolus_out = min(extra / 1000 * self.T, self.max_bolus)  # cap bolus

        # safety guardrail
        if current_cgm < 100 or self.pw_data_object.get_dcgm_dt()/self.T < -3:  # dcgm_dt in mg/dL per minute
            basal_out = 0.0
            bolus_out = 0.0

        return basal_out, bolus_out

    def mpc_set(self):

        nx = self.linear_model.get_state_num()
        nu = self.linear_model.get_input_num_artificial()
        ny = self.linear_model.get_output_num()

        eq_constraints_num = ny + ny + nx

        # decision variables are: y, e, x, u
        sample_time_decision_var_num = ny + ny + nx + nu

        weight_matrix = np.zeros((sample_time_decision_var_num, sample_time_decision_var_num))
        weight_matrix[ny:ny + ny, ny:ny + ny] = self.mpc_obj_fcn_params.Q_e  # e

        H = np.kron(np.eye(self.prediction_horizon), weight_matrix)
        f = np.zeros((sample_time_decision_var_num * self.prediction_horizon, 1))

        A_inequality = None
        b_inequality = None

        # equality constraints
        A0_equality = np.concatenate([
            np.concatenate([[1], np.zeros(ny + nx + nu)]).reshape(1, -1),  # y
            np.concatenate([[1], [-1], np.zeros(nx + nu)]).reshape(1, -1),  # e
            np.concatenate([np.zeros((nx, 2 * ny)), np.eye(nx), np.zeros((nx, nu))], axis=1),  # x
        ], axis=0)

        A0_equality = np.concatenate(
            [A0_equality, np.zeros((A0_equality.shape[0], sample_time_decision_var_num * (self.prediction_horizon - 1)))],
            axis=1)

        A11_equality_block = np.concatenate([
            np.zeros((ny + ny, sample_time_decision_var_num)),  # y, e
            np.concatenate([np.zeros((nx, 2 * ny)), self.linear_model.get_A(), self.linear_model.get_B_artificial()], axis=1),
            # x
        ], axis=0)

        A11_equality = np.kron(np.eye(self.prediction_horizon - 1), A11_equality_block)
        A11_equality = np.concatenate(
            [A11_equality, np.zeros(((self.prediction_horizon - 1) * eq_constraints_num, sample_time_decision_var_num))],
            axis=1)

        A12_equality_block = np.concatenate([
            np.concatenate([[1], np.zeros(ny), -self.linear_model.get_C().reshape(-1), np.zeros(nu)]).reshape(1, -1),  # y
            np.concatenate([[1], [-1], np.zeros(nx + nu)]).reshape(1, -1),  # e
            np.concatenate([np.zeros((nx, 2 * ny)), -np.eye(nx), np.zeros((nx, nu))], axis=1),  # x
        ], axis=0)

        A12_equality = np.kron(np.eye(self.prediction_horizon - 1), A12_equality_block)
        A12_equality = np.concatenate(
            [np.zeros(((self.prediction_horizon - 1) * eq_constraints_num, sample_time_decision_var_num)), A12_equality],
            axis=1)
        A1_equality = A11_equality + A12_equality
        A_equality = np.concatenate([A0_equality, A1_equality], axis=0)

        b0_equality = np.concatenate([
            [0,  # y
             self.mpc_obj_fcn_params.ysp],  # e
            np.zeros(nx),  # x
        ]).reshape(-1, 1)

        b1_equality_block = np.concatenate([
            [0,  # y
             self.mpc_obj_fcn_params.ysp],  # e
            np.zeros(nx),  # x
        ]).reshape(-1, 1)

        b1_equality = np.tile(b1_equality_block, (self.prediction_horizon - 1, 1))
        b_equality = np.concatenate([b0_equality, b1_equality], axis=0)

        first_input_indices = np.arange(sample_time_decision_var_num - nu,
                                        sample_time_decision_var_num * self.prediction_horizon, sample_time_decision_var_num)
        u_index = np.hstack(
            [first_input_indices, first_input_indices + 1, first_input_indices + 2, first_input_indices + 3])
        u_index = np.sort(u_index)

        lb_ = np.concatenate([self.bounds.y_min, np.full(ny, -np.inf), self.bounds.x_min, np.zeros(nu)]).reshape(-1, 1)
        lb = np.tile(lb_, (self.prediction_horizon, 1))
        lb[u_index] = self.bounds.u_min.T.reshape(-1, 1)

        ub_ = np.concatenate([self.bounds.y_max, np.full(ny, np.inf), self.bounds.x_max, np.zeros(nu)]).reshape(-1, 1)
        ub = np.tile(ub_, (self.prediction_horizon, 1))
        ub[u_index] = self.bounds.u_max.T.reshape(-1, 1)

        x0_ = np.concatenate([[0], [0], np.zeros(nx), np.zeros(nu)]).reshape(-1, 1)
        x0 = np.tile(x0_, (self.prediction_horizon, 1))

        qop = Quadratic_Optimization_Problem(H=H, f=f, A=A_inequality, b=b_inequality, Aeq=A_equality, beq=b_equality,
                                             lb=lb, ub=ub, x0=x0)

        return qop

    @staticmethod
    def mpc_solve(qop: Quadratic_Optimization_Problem):

        # try:
        x_solution = qpsolvers.solve_qp(P=qop.H, q=qop.f, G=qop.A, h=qop.b, A=qop.Aeq, b=qop.beq, lb=qop.lb,
                                        ub=qop.ub, solver='clarabel')
        exit_flag = True
        # except qpsolvers.exceptions.ProblemError:
        #     x_solution = None
        #     exit_flag = False
        # except Exception:
        #     x_solution = None
        #     exit_flag = False

        return x_solution, exit_flag

    @staticmethod
    def get_insulin_input_mU_min(basal_val, bolus_val):
        return basal_val * 1000 / 60 + bolus_val * 1000

    def state_space_initializer(self):
        # T is the sampling rate of CGM values in minutes

        TmaxI = self.hovorka_params.TmaxI
        Ke = self.hovorka_params.Ke
        K12 = self.hovorka_params.K12
        EGP0 = self.hovorka_params.EGP0
        BW = self.hovorka_params.BW
        gain = self.hovorka_params.gain
        TmaxE = self.hovorka_params.TmaxE
        ka1 = self.hovorka_params.ka1
        ka2 = self.hovorka_params.ka2
        ka3 = self.hovorka_params.ka3
        kb1 = self.hovorka_params.get_kb1()
        kb2 = self.hovorka_params.get_kb2()
        kb3 = self.hovorka_params.get_kb3()
        VI = self.hovorka_params.get_VI()
        VG = self.hovorka_params.get_VG()
        F01 = self.hovorka_params.F01
        Ag = self.hovorka_params.Ag
        TmaxG = self.hovorka_params.TmaxG

        Q2 = self.current_states_object.Q2
        R2 = self.current_states_object.R2
        x1 = self.current_states_object.x1
        x2 = self.current_states_object.x2

        G = self.pw_data_object.get_last_cgm() / 18.0
        sw1 = 1 if G >= 4.5 else 0
        sw2 = 1 if G >= 9.0 else 0

        # p = [0.0053, 0.0257] # 2 parameter exercise model
        p = [0, 0.1253]  # 1 parameter exercise model
        alpha = p[0]
        beta = p[1]

        Ac = np.array([
            [-1 / TmaxI, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / TmaxI, -1 / TmaxI, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1 / (VI * TmaxI), -Ke, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -(1 + alpha * R2) * x1 - (1 - sw1) * F01 * BW / (4.5 * VG) - sw2 * 0.003, K12 / VG, 0,
             -alpha * x1 * G, -(1 + alpha * R2) * G, 0, -BW * EGP0 / VG, 0, 5.556 / (TmaxG * VG)],
            [0, 0, 0, (1 + alpha * R2) * x1 * VG, -K12 - (1 + beta * R2) * x2, 0, alpha * x1 * G * VG - beta * x2 * Q2,
             (1 + alpha * R2) * G * VG, -(1 + beta * R2) * Q2, 0, 0, 0],
            [0, 0, 0, 0, 0, -1 / TmaxE, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1 / TmaxE, -1 / TmaxE, 0, 0, 0, 0, 0],
            [0, 0, kb1, 0, 0, 0, 0, -ka1, 0, 0, 0, 0],
            [0, 0, kb2, 0, 0, 0, 0, 0, -ka2, 0, 0, 0],
            [0, 0, kb3, 0, 0, 0, 0, 0, 0, -ka3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / TmaxG, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / TmaxG, -1 / TmaxG]
        ])

        Bc = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, gain],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, Ag, 0],
            [0, 0, 0]
        ])

        num_states = Ac.shape[0]
        num_inputs = Bc.shape[1]
        output_state_pos = 4

        A = np.eye(num_states) + self.T * Ac
        B = self.T * Bc
        C = np.array([np.concatenate((np.zeros(output_state_pos - 1), [1], np.zeros(num_states - output_state_pos)))])
        D = np.zeros((1, num_inputs))

        return A, B, C, D

    def get_hovorka_model_f0(self, x, u):
        ka1 = self.hovorka_params.ka1
        ka2 = self.hovorka_params.ka2
        ka3 = self.hovorka_params.ka3
        TmaxI = self.hovorka_params.TmaxI
        SIT = self.hovorka_params.SIT
        SID = self.hovorka_params.SID
        SIE = self.hovorka_params.SIE
        Ke = self.hovorka_params.Ke
        K12 = self.hovorka_params.K12
        gain = self.hovorka_params.gain
        TmaxE = self.hovorka_params.TmaxE
        EGP0 = self.hovorka_params.EGP0
        BW = self.hovorka_params.BW
        F01 = self.hovorka_params.F01
        Ag = self.hovorka_params.Ag
        TmaxG = self.hovorka_params.TmaxG

        VI = self.hovorka_params.get_VI()
        VG = self.hovorka_params.get_VG()

        m = 3  # the fourth state in x is G
        if x[m] >= 4.5:
            F01C = F01 * BW
        else:
            F01C = F01 * BW * x[m] / 4.5

        if x[m] >= 9:
            FR = 0.003 * (x[m] - 9) * VG
        else:
            FR = 0

        EE_rest = 1

        # double parameter exercise compartment
        # alpha = 0.0053
        # beta = 0.0257

        # single parameter exercise compartment
        alpha = 0
        beta = 0.1253

        f = np.zeros(len(x))

        f[0] = u[0] - x[0] / TmaxI
        f[1] = x[0] / TmaxI - x[1] / TmaxI
        f[2] = x[1] / (TmaxI * VI) - Ke * x[2]

        f[3] = (1 / VG) * (x[11] * 5.556 / TmaxG - F01C - FR - (1 + alpha * x[6]) * x[7] * x[3] * VG + K12 * x[
            4] + BW * EGP0 * (1 - x[9]))
        f[4] = x[3] * VG * (1 + alpha * x[6]) * x[7] - (K12 + (1 + beta * x[6]) * x[8]) * x[4]

        f[5] = (gain * (u[2] - EE_rest) - x[5] / TmaxE)
        f[6] = (x[5] / TmaxE - x[6] / TmaxE)

        f[7] = -ka1 * x[7] + SIT * ka1 * x[2]
        f[8] = -ka2 * x[8] + SID * ka2 * x[2]
        f[9] = -ka3 * x[9] + SIE * ka3 * x[2]

        f[10] = Ag * u[1] - x[10] / TmaxG
        f[11] = x[10] / TmaxG - x[11] / TmaxG

        return f

    def update_linear_model(self):
        A, B, C, D = self.state_space_initializer()
        self.linear_model.A = A
        self.linear_model.B = B
        self.linear_model.C = C
        self.linear_model.D = D
        self.linear_model.x0 = self.current_states_object.get_short_state_vector()
        self.linear_model.f0 = self.get_hovorka_model_f0(self.current_states_object.get_short_state_vector(),
                                                         self.get_u0())

    def get_u0(self):
        if self.pw_data_object.push_num == 1:
            u0 = np.array([UnitConversion.insulin.Uhr_to_uUmin(self.demographic_info.basal_rate), 0, 1])
        else:
            u0 = np.array([self.pw_data_object.get_last_insulin_input_mU_min(),
                                       self.pw_data_object.meal_pw[-2], self.pw_data_object.energy_expenditure_pw[-2]])

        return u0

    def update_state(self, x_plant):
        # this method is not called for MPC classes that possess an internal observer
        # since this class does not run an observer, calling this method updates the state of MPC by x_plant argument
        # from plant
        self.current_states_object.update_by_vector(x_plant)

    def run(self, measurements, inputs, states, sample, patient_idx):
        patient_states = states[patient_idx,:,sample]

        # Convert representation from simulator to controller
        #[S1, S2, PIC, G, Q2, R1, R2, x1, x2, x3, D1, D2]
        #['S1', 'S2', 'I', 'x1', 'x2', 'x3', 'Q1', 'Q2', 'IG', 'D1Slow', 'D2Slow','D1Fast', 'D2Fast', 'EEfast', 'EEhighintensity', 'EElongeffect']
        converted_states = np.zeros(12,)
        converted_states[0:3] = patient_states[0:3]
        converted_states[3] = patient_states[8]
        converted_states[4] = patient_states[7]
        converted_states[[5,6]] = [0,0]
        converted_states[7:10] = patient_states[3:6]
        converted_states[10:] = UnitConversion.glucose.mmol_glucose_to_g(patient_states[9:11])#patient_states[9:11]
        self.update_state(converted_states)
        # Select hardcoded 1st patient (MPC currently works for single patient)
        G = UnitConversion.glucose.concentration_mmolL_to_mgdL(measurements[patient_idx])
        uFastCarbs, uSlowCarbs, uHR, uInsulin, energy_expenditure = inputs[patient_idx]

        sum_meals = np.sum(uFastCarbs[sample-self.T+1:sample:sample+1]+uSlowCarbs[sample-self.T+1:sample:sample+1])/self.T
        energy_expenditure_mean = 1+np.sum(energy_expenditure[sample-self.T+1:sample:sample+1])/self.T


        # meal is in gram
        # exercise is in MET
        self.pw_data_object.push_cgm(G)
        self.pw_data_object.push_meal(sum_meals)
        self.pw_data_object.push_energy_expenditure(energy_expenditure_mean)

        self.update_linear_model()
        basal, bolus = self.mpc_execute()

        # inject insulin related faults

        self.pw_data_object.push_basal(basal)
        self.pw_data_object.push_bolus(bolus)
        # insulin = MPC_Controller_01.get_insulin_input_mU_min(basal, bolus)
        insulin_rate = UnitConversion.insulin.Uhr_to_mUmin(basal)+UnitConversion.insulin.U_to_mU(bolus)/self.T
        # basal_rate = UnitConversion.insulin.Uhr_to_mUmin(basal)
        # bolus_rate = UnitConversion.insulin.U_to_mU(bolus)/self.T
        inputs[patient_idx,3,sample:sample+self.T] = insulin_rate
        return
