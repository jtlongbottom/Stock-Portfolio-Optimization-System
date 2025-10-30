import streamlit as st

st.title("Risk Tolerance Quiz")
risk_score = 0

Q1 = "What is your invesment goal?"
O1_1 = "Retirement"
O1_2 = "General Investing (i.e. Saving for house or car)"
O1_3 = "Wealth Growth"

O1 = [O1_1,O1_2,O1_3]

A1 = st.selectbox(Q1,O1, index=None)

if A1 == O1_1:
    Q2 = "What is your age?"
    A2 = st.number_input(Q2, min_value=1, step=1, format="%d", value=None)
    Q3 = "What age do you plan to retire?"
    A3 = st.number_input(Q3, min_value=1, step=1, format="%d", value = 67)
    if A2:
        timeline = A3 - A2
        if timeline <= 2:
            risk_score += 1
        elif timeline <= 5:
            risk_score += 2
        elif timeline <= 10:
            risk_score += 4
        else:
            risk_score += 5
elif A1 != None: 
    Q4 = "What is your investment timeframe?"
    O4_1 = "0-2 years"
    O4_2 = "2-5 years"
    O4_3 = "5-10 years"
    O4_4 = "10+ years"
    O4 = [O4_1,O4_2,O4_3,O4_4]
    A4 = st.selectbox(Q4, O4, index=None)
    if A4:
        if A4 == O4_1:
            risk_score +=1
        elif A4 == O4_2:
            risk_score += 2
        elif A4 == O4_3:
            risk_score += 4
        elif A4 == O4_4:
            risk_score += 5



Q5 = "What expected returns are you hoping for?"
O5_1 = "2-4%"
O5_2 = "5-8%"
O5_3 = "9+%"
O5 = [O5_1, O5_2, O5_3]

A5 = st.radio(Q5,O5, index=None)

if A5:
    if A5 == O5_1:
        risk_score +=1
    elif A5 == O5_2:
        risk_score += 3
    elif A5 == O5_3:
        risk_score += 5

Q6 = "What is investment management style?"
O6_1 = "I want to invest and forget about it until I need it"
O6_2 = "I like to make periodic checks, but not too frequently"
O6_3 = "I enjoy monitoring my portfolio and making active decisions"
O6 = [O6_1, O6_2, O6_3]

A6 = st.radio (Q6,O6, index=None)

if A6:
    if A6 == O6_1:
        risk_score +=1
    elif A6 == O6_2:
        risk_score += 3
    elif A6 == O6_3:
        risk_score += 5

Q7 = "You have a portfolio of \$100,000 and it drops to \$80,000 what do you do?"
O7_1 = "Sell to prevent further loss"
O7_2 = "Hold out because the market will bounce back"
O7_3 = "Invest more while the prices are lower"
O7 = [O7_1, O7_2, O7_3]
A7 = st.radio(Q7,O7,index=None)

if A7:
    if A7 == O7_1:
        risk_score +=1
    elif A7 == O7_2:
        risk_score += 3
    elif A7 == O7_3:
        risk_score += 5

Q8 = "How comfortable are you with personal wealth management?"
O8_1 = "Not comfortable at all"
O8_2 = "Somewhat comfortable"
O8_3 = "Generally comfortable"
O8_4 = "Extremely comfortable"
O8 = [O8_1, O8_2, O8_3, O8_4]

A8 = st.radio(Q8,O8, index=None)

if A8:
    if A8 == O8_1:
        risk_score +=1
    elif A8 == O8_2:
        risk_score += 2
    elif A8 == O8_3:
        risk_score += 4
    elif A8 == O8_4:
        risk_score += 5

if st.button("Submit"):
    st.write("Your risk score is:", risk_score)
    tolerances = ["Conservative","Moderate","Aggressive"]
    if risk_score<10:
        st.success(f"Your risk tolerance is estimated to be: {tolerances[0]}")
    elif risk_score<18:
        st.success(f"Your risk tolerance is estimated to be: {tolerances[1]}")
    else:
        st.success(f"Your risk tolerance is estimated to be: {tolerances[2]}")

