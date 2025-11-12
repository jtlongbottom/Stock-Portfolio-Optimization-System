import streamlit as st
import pandas as pd
import plotly.express as px
from helper import get_stock_names
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
#from Gather_Data import main_df
cached_stock_names = st.cache_data(get_stock_names, ttl = 3600)

#def get_data():
#    return Gather_Data.

st.set_page_config(page_title="Stock Portfolio Optimization")

st.title("Stock Portfolio Optimization")







### Tolerance ###

tolerance = st.sidebar.selectbox("Tolerance:",["Conservative","Moderate","Aggressive"])
st.sidebar.markdown('<p style="font-size:13px; color:white;">What is my tolerance? Check out Risk Tolerance Questionaire Page Above</p>', unsafe_allow_html=True)


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


### Placeholder Chart ###

def create_df():
    dates = pd.date_range(start="2023-01-01",end="2025-10-28", freq="D")
    n=len(dates)
    price = [100]
    for _ in range(1, n):
        change = 1 + 0.001 + 0.02 * np.random.randn()
        price.append(price[-1] * change)
    #trend = np.cumsum(np.random.rand(n)*0.5+0.1**n)
    df = pl.DataFrame({"Date":pl.Series("Date", dates),
                       "Price": pl.Series("Price",price, dtype=pl.Float64)})
    return df



def create_graph(df):
    #theme = st.get_option("theme.base")
    query_params = st.query_params
    theme = query_params.get("theme",["dark"])[0]
    #print(theme)
    #default is always dark could add button to switch between dark and light
    if theme == "dark":
        bg_color = "#0E1117"
        grid_color = "#555555"
        text_color = "white"
        line_color = "cyan"
    else:
        bg_color = "white"
        grid_color = "#cccccc"
        text_color = "black"
        line_color = "teal"

    fig, ax = plt.subplots(figsize=(10,5), facecolor = bg_color)
    ax.plot(df["Date"], df["Price"], color = line_color, linewidth=2)
    ax.set_title("Title", color = text_color)
    ax.set_xlabel("Dates", color = text_color)
    ax.set_ylabel("Price", color = text_color)
    ax.tick_params(axis="x", color = text_color)
    ax.tick_params(axis="y", color = text_color)

    for label in ax.get_xticklabels():
        label.set_color(text_color)
    for label in ax.get_yticklabels():
        label.set_color(text_color)

    ax.set_facecolor(bg_color)
    ax.grid(True, alpha=0.3, color = grid_color)
    return fig

df1 = create_df()
df2 = create_df()

spos = create_graph(df1)
sp500 = create_graph(df2)


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

pie = px.pie(invest_pie, names ='Investment',values='Amount', 
             color = 'Investment',
             category_orders={'Investment':invest_order}, 
             color_discrete_map=color_map,  
             template="plotly_white",
             title = 'Percentages')
pie.update_layout(width=400, height=400, 
                  legend=dict(orientation="h"),
                  margin=dict(t=100,b=10,l=10,r=10))


### metrics ###

exp_return = 0.12
std_dev = 0.04
bond = 0.03
sharpe_ratio = (exp_return - bond)/std_dev





### Main plot###
col1, col2 = st.columns([2,1])
with col1:
    st.write("Main stuff")
    show_fan = st.checkbox("Display Fan Chart", value = False)
    tab= st.radio("", ["SPOS","S&P 500"], key="chart_tab")
    if tab == "SPOS":
        st.pyplot(spos)
    if tab == "S&P 500":
        st.pyplot(sp500)
with col2:
    st.plotly_chart(pie, use_container_width=False, theme=None)
    if tab == "SPOS":
        st.metric("Expected Return", f"{ exp_return *100:.2f}%")
        st.metric("Volatility", f"{ std_dev *100:.2f}%")
        st.metric("Sharpe Ratio", f"{ sharpe_ratio:.2f}")
    elif tab == "S&P 500": 
        st.metric("Expected Return", f"{ exp_return *100+10:.2f}%", delta = f"{10}%")
        st.metric("Volatility", f"{ std_dev *100-1:.2f}%", delta = f"{-1}%")
        st.metric("Sharpe Ratio", f"{ sharpe_ratio+0.1:.2f}", delta = 0.1)


#### Stock Selection #######

df = pd.read_csv('top1000_assets_ranked_by_sharpe_ratio.csv')
tickers = df['asset'][:200]

stocks = cached_stock_names(tickers)

selected_stocks = st.sidebar.multiselect("Select Stocks", stocks, max_selections=10)
if selected_stocks:
    st.sidebar.write(selected_stocks)



