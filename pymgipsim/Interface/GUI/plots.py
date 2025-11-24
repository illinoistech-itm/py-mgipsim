import streamlit as st
from bokeh import plotting
from bokeh import models
from bokeh.colors import RGB
from bokeh.models import Arrow, VeeHead
from bokeh.palettes import Category20
import numpy as np
from pymgipsim.Utilities.units_conversions_constants import UnitConversion
from pymgipsim.Controllers import *
from pymgipsim.VirtualPatient.Models import T1DM

def plot_hovorka():
    patient_list = [file.replace(".json","") for file in st.session_state.simulated_scenario.patient.files]
    options = st.multiselect(
        "Select patients",
        patient_list,default=patient_list[0])

    # Build available variables list - IOB only available for ExtHovorka + OpenAPS
    available_vars = ["Blood glucose","Meals","Insulin","Heart rate"]
    scenario = st.session_state.simulated_scenario
    model = st.session_state.model
    if (model is not None and
        model.name == T1DM.ExtHovorka.Model.name and
        scenario.controller.name == OpenAPS.controller.Controller.name):
        available_vars.insert(3, "IOB")  # Insert IOB after Insulin

    variables = st.multiselect(
        "Select variables",
        available_vars,default="Blood glucose")
    if st.session_state.model is not None:
        cho_color = np.asarray([0.259, 0.125, 0.329])*255
        insulin_color =  np.asarray([0, 0.639, 0.224])*255
        iob_color = np.asarray([0.2, 0.4, 0.6])*255  # Blue color for IOB
        heart_rate_color = np.asarray([1, 0.498, 0.545, 0.25/255])*255  # [1, 0.325, 0.392]

        model = st.session_state.model
        p = plotting.figure(title="T1DM cohort results.", x_axis_label='x', y_axis_label='y')
        time_axis = model.time.as_unix / 60.0
        selected_patients = [patient_list.index(option) for option in options]
        color = Category20[max(3, model.states.as_array.shape[0])]
        for patientidx in selected_patients:
            glucose = UnitConversion.glucose.concentration_mmolL_to_mgdL(
                model.states.as_array[patientidx, model.glucose_state, :] / model.parameters.VG[
                    patientidx])
            if "Blood glucose" in variables:
                p.line(time_axis, glucose, legend_label=patient_list[patientidx], line_width=2,
                       color=color[patientidx])
            p.y_range = models.Range1d(0, 400)
            for magnitude, start_time in zip(scenario.inputs.meal_carb.magnitude[patientidx],
                                             scenario.inputs.meal_carb.start_time[patientidx]):
                vh = VeeHead(size=8, fill_color=RGB(*cho_color))
                if "Meals" in variables:
                    p.add_layout(Arrow(end=vh, x_start=start_time / 60.0, y_start=0, x_end=start_time / 60.0, y_end=magnitude))

            if "Insulin" in variables:
                try:
                    if scenario.controller.name == OpenLoop.controller.Controller.name or scenario.controller.name == StochasticOpenLoop.controller.Controller.name:
                        for magnitude, start_time in zip(scenario.inputs.bolus_insulin.magnitude[patientidx],
                                                         scenario.inputs.bolus_insulin.start_time[patientidx]):
                            vh = VeeHead(size=8, fill_color=RGB(*insulin_color))
                            p.add_layout(Arrow(end=vh, x_start=start_time / 60.0, y_start=0, x_end=start_time / 60.0,
                                               y_end=magnitude, line_color=RGB(*insulin_color)))
                    else:
                        match model.name:
                            case T1DM.ExtHovorka.Model.name:
                                insulin_arr = UnitConversion.insulin.mUmin_to_Uhr(model.inputs.as_array[patientidx, 3, :])
                                p.varea(x=time_axis,
                                        y1=np.zeros_like(insulin_arr),
                                        y2=insulin_arr,color=RGB(*insulin_color))
                            case T1DM.IVP.Model.name:
                                insulin_arr = UnitConversion.insulin.uUmin_to_Uhr(model.inputs.as_array[patientidx, 1, :])
                                p.varea(x=time_axis,
                                        y1=np.zeros_like(insulin_arr),
                                        y2=insulin_arr,color=RGB(*insulin_color))
                except:
                    pass

            # Plot IOB (only for ExtHovorka with OpenAPS)
            if "IOB" in variables:
                try:
                    if (model.name == T1DM.ExtHovorka.Model.name and
                        scenario.controller.name == OpenAPS.controller.Controller.name):
                        iob_arr = model.inputs.as_array[patientidx, 5, :]  # IOB is at index 5
                        p.line(time_axis, iob_arr, legend_label="IOB [U]", line_width=2, color=RGB(*iob_color))
                except:
                    pass

            try:
                if "Heart rate" in variables:
                        hr_times = scenario.inputs.heart_rate.start_time[patientidx]
                        hr_magnitudes = scenario.inputs.heart_rate.magnitude[patientidx]
                        hr_times = np.asarray(hr_times) / 60
                        hr_magnitudes = np.asarray(hr_magnitudes)
                        hr_times = np.append(hr_times, time_axis[-1])
                        hr_magnitudes = np.append(hr_magnitudes, hr_magnitudes[-1])
                        p.varea(x=hr_times,
                                y1=np.zeros_like(hr_magnitudes),
                                y2=hr_magnitudes, color=RGB(*heart_rate_color))
            except:
                pass


        st.bokeh_chart(p, use_container_width=True)


# def plot_jauslin():
    # if st.session_state.model is not None and st.session_state.model.name == T2DM.Jauslin.Model.name:
    #     model = st.session_state.model
    #     p = plotting.figure(title="T2DM cohort blood glucose results.", x_axis_label='x', y_axis_label='y')
    #     time_axis = model.time.as_unix / 60.0
    #
    #     glucose = np.zeros_like(model.states.as_array[:, model.glucose_state, :])
    #     volume = model.parameters.Vg
    #     for i in range(glucose.shape[0]):
    #         glucose[i] = UnitConversion.glucose.concentration_mmolL_to_mgdL(model.states.as_array[i, model.glucose_state, :] / volume[i])
    #     glucose_std = glucose.std(axis=0)
    #     glucose_mean = glucose.mean(axis=0)
    #
    #     p.line(time_axis, glucose_mean, legend_label="Cohort avg. glucose trace.", line_width=2)
    #
    #     p.varea(x=time_axis,
    #             y1=glucose_mean-glucose_std,
    #             y2=glucose_mean+glucose_std, alpha=0.25)
    #
    #     st.bokeh_chart(p, use_container_width=True)


def plot_multiscale():
    if st.session_state.multiscale_model is not None:
        model = st.session_state.multiscale_model
        bw_state = model.states.as_array
        time_axis = model.time.as_unix
        p = plotting.figure(title="Body weight change", x_axis_label='x', y_axis_label='y')

        mean = bw_state.mean(axis=0)[0]
        std = bw_state.std(axis=0)[0]

        p.line(time_axis, mean, legend_label="Cohort avg. glucose trace.", line_width=2)

        p.varea(x=time_axis,
                y1=mean-std,
                y2=mean+std, alpha=0.25)

        st.bokeh_chart(p, use_container_width=True)