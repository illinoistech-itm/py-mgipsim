import streamlit as st
from pymgipsim.VirtualPatient.Models import *
from pymgipsim.Controllers import *


def therapies():
    st.header("Therapies")

    if st.session_state.args.model_name == T1DM.ExtHovorka.Model.name:
        option = st.selectbox(
            "Insulin therapy options",
            (
                "💉 Multiple daily injections",
                "💉 Multiple daily injections (stochastic)",
                "📱 Sensor augmented pump therapy",
                "📱 Hybrid closed-loop",
                "📱 OpenAPS",
            ),
        )
        if "💉 Multiple daily injections" == option:
            st.session_state.args.controller_name = OpenLoop.controller.Controller.name
        if "💉 Multiple daily injections (stochastic)" == option:
            st.session_state.args.controller_name = (
                StochasticOpenLoop.controller.Controller.name
            )
        if "📱 Sensor augmented pump therapy" == option:
            st.session_state.args.controller_name = SAPT.controller.Controller.name
        if "📱 Fully automated insulin delivery" == option:
            st.session_state.args.controller_name = ""
        if "📱 Hybrid closed-loop" == option:
            st.session_state.args.controller_name = HCL0.controller.Controller.name
        if "📱 OpenAPS" == option:
            st.session_state.args.controller_name = OpenAPS.controller.Controller.name
    else:
        st.session_state.args.controller_name = "OpenLoop"

    # if st.session_state.args.model_name == T2DM.Jauslin.Model.name:
    #     st.session_state.args.sglt2i_dose_magnitude = st.number_input(
    #         "💊 SGLT2i dose magnitude", value=0,
    #         min_value=0, max_value=10, step=1
    #     )
