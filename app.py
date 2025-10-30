import streamlit as st
st.set_page_config(page_title="Stock Portfolio Optimization")

st.title("Stock Portfolio Optimization")



tolerance = st.sidebar.selectbox("Tolerance:",["Conservative","Moderate","Aggressive"])
st.sidebar.markdown('<p style="font-size:13px; color:white;">What is my tolerance? Check out Risk Tolerance Questionaire Page</p>', unsafe_allow_html=True)

if "range" not in st.session_state:
    st.session_state.range = [1990,2025]


min_val, max_val = st.sidebar.slider("Years:",
                        1990,
                        2025,
                        tuple(st.session_state.range),
                        step=1,
                        key = "range_slider")

min_input = st.sidebar.number_input("Beginning", 
                        1990, 
                        2025, 
                        min_val,
                        step=1)
max_input = st.sidebar.number_input("Ending", 
                        1990, 
                        2025, 
                        max_val,
                        step=1)

st.session_state.range = [min_input, max_input]
