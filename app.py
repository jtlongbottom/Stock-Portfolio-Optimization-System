import streamlit as st
import pandas as pd
import plotly.express as px
from helper import get_stock_names
cached_stock_names = st.cache_data(get_stock_names, ttl = 3600)


st.set_page_config(page_title="Stock Portfolio Optimization")

st.title("Stock Portfolio Optimization")







### Tolerance ###

tolerance = st.sidebar.selectbox("Tolerance:",["Conservative","Moderate","Aggressive"])
st.sidebar.markdown('<p style="font-size:13px; color:white;">What is my tolerance? Check out Risk Tolerance Questionaire Page</p>', unsafe_allow_html=True)


default_percentages = {
    "Conservative": [20,50,30],
    "Moderate": [60,35,5],
    "Aggressive": [80,15,5]
}


for key in ["bonds_percent","stock_percent","cash_percent","last_tolerance"]:
    if key not in st.session_state:
        st.session_state[key] = 0

if st.session_state.get("last_tolerance") != tolerance:
    st.session_state.bonds_percent = default_percentages[tolerance][0]
    st.session_state.stock_percent = default_percentages[tolerance][1]
    st.session_state.cash_percent = default_percentages[tolerance][2]
    st.session_state.last_tolerance = tolerance

bonds_percent = st.sidebar.number_input("Bonds Percentage", 
                        0, 
                        100,
                        step=1,
                        value = st.session_state.bonds_percent,
                        key="bonds_percent")
stock_percent = st.sidebar.number_input("Stock Percentage", 
                        0, 
                        100,
                        step=1,
                        value = st.session_state.stock_percent,
                        key="stock_percent")
cash_percent = 100-bonds_percent-stock_percent

if cash_percent <0:
    st.error("Sum of Bonds and Stocks is greater than 100%")
    cash_percent = 0

st.sidebar.markdown(f"""
                    <div style="
                    font-family: 'Sans-Serif';
                    font-size: 16px"> Cash Percent (Auto):
                    {cash_percent}</div>""",
                    unsafe_allow_html=True)
st.sidebar.write("")

#might want to add a greyed out portion to tolerance select menu to signify a change has been made


percentages = [bonds_percent,stock_percent,cash_percent]






### Years ###

if False:
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






### Pie Chart ###


color_map = {
    "Bonds": "#ebe124",
    "Stock": "#2c32a0",
    "Cash Equivalents": "#d62727"
}


invest_order = ["Bonds","Stock","Cash Equivalents"]
invest_pie = pd.DataFrame({'Investment':["Bonds","Stock","Cash Equivalents"],
                    'Amount':percentages})

invest_pie = (invest_pie.set_index('Investment').reindex(invest_order, fill_value=0).reset_index())

fig = px.pie(invest_pie, names ='Investment',values='Amount', 
             color = 'Investment',
             category_orders={'Investment':invest_order}, 
             color_discrete_map=color_map,  
             template="plotly_white",
             title = 'Percentages')
fig.update_layout(width=400, height=400, 
                  legend=dict(orientation="h"),
                  margin=dict(t=100,b=10,l=10,r=10))

col1, col2 = st.columns([2,1])
with col1:
    st.write("Main stuff")
with col2:
    st.plotly_chart(fig, use_container_width=False, theme=None)



#### Stock Selection #######

df = pd.read_csv('top1000_assets_ranked_by_sharpe_ratio.csv')
tickers = df['asset'][:200]

stocks = cached_stock_names(tickers)

selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, max_selections=10)
if selected_stocks:
    st.sidebar.write(selected_stocks)



