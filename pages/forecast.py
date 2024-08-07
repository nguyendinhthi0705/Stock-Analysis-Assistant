# import yfinance as yf
# import streamlit as st
# import time
# import pandas as pd
# from streamlit_option_menu import option_menu
# import numpy as np
# from datetime import date,datetime,timedelta
# from plotly import graph_objs as go
# from prophet import Prophet
 

# st.header("Forecasting")
# st.write("---")
# n_years = st.slider("Months Of Prediction: ", 1, 24)
# period = n_years * 30

# snp500 = pd.read_csv("SP500.csv")
# symbols = snp500['Symbol'].sort_values().tolist()    

# ticker = st.sidebar.selectbox('Chọn mã chứng khoán', symbols)
# # Preprocess the data
# data = yf.download(ticker,start="2024-01-01", end=date.today().strftime("%Y-%m-%d"))
# df = data.reset_index()[["Date", "Close"]].copy()
# df["Date"] = pd.to_datetime(df["Date"])
# df["DayOfWeek"] = df["Date"].dt.dayofweek
# df = df[(df["DayOfWeek"] != 5) & (df["DayOfWeek"] != 6)]
# df = df.rename(columns={"Date": "ds", "Close": "y"})
            
# # Train the Prophet model
# m = Prophet(daily_seasonality=True, yearly_seasonality=True)
# m.fit(df)
#  # Make future dataframe and predictions
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)
# forecast1 = forecast.rename(columns={'ds':'Date','yhat':'Predicted Prices','yhat_lower':'Predicted Lowest','yhat_upper':'Predicted Highest'})
# forelist = ['Date', 'Predicted Prices', 'Predicted Lowest', 'Predicted Highest']
# print(forecast)            
# st.write("---")
# st.subheader("Forecast Data")
# st.write(forecast1[forelist])
# st.write('---')
            
# # Prediction interval
# st.subheader("Prediction in Interval of Time")
# Start = st.date_input('Enter start date', value=None)
# End = st.date_input('Enter end date', value=None)
# if Start != End:
#     selected_forecast = forecast1.loc[(forecast1['Date'] > pd.to_datetime(Start)) & (forecast1['Date'] <= pd.to_datetime(End))]
#     st.write(selected_forecast[forelist])
# st.write("---")
            
# # Forecasted Data Graphs
# st.subheader('Forecasted Data Graphs')
# st.write("Actual Prices v/s Predicted Prices")
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Prices', line=dict(color='red')))
# fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Prices', marker=dict(color='green')))
# fig.update_layout(xaxis_rangeslider_visible=False)
# st.plotly_chart(fig)
            
# st.write("Forecast Components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

#             # Create the candlestick chart
# fig = go.Figure(data=[
#     go.Candlestick(
#         x=df['ds'],
#         open=df['open'],
#         high=df['high'],
#         low=df['low'],
#         close=df['close'],
#         name='Actual Prices'
#     ),
#     go.Scatter(
#         x=forecast['ds'],
#         y=forecast['yhat'],
#         mode='lines',
#         name='Predicted Prices',
#         line=dict(color='red')
#     )
# ])
# fig.update_layout(xaxis_rangeslider_visible=False)
# st.plotly_chart(fig)
# st.button("Exit")
# st.write("---")                       


import yfinance as yf
import streamlit as st
import time
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
from datetime import date,datetime,timedelta
from plotly import graph_objs as go
from prophet import Prophet
import boto3, json


st.header("Forecasting")
st.write("---")
n_years = st.slider("Months Of Prediction: ", 1, 24)
period = n_years * 30

snp500 = pd.read_csv("SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()    

ticker = st.sidebar.selectbox('Chọn mã chứng khoán', symbols)

# Model AI
def call_claude_sonet_stream(prompt):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0, 
        "top_k": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    bedrock = boto3.client(service_name="bedrock-runtime")  
    response = bedrock.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    stream = response['body']
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                 delta = json.loads(chunk.get('bytes').decode()).get("delta")
                 if delta:
                     yield delta.get("text")

def forecast_price(question, docs): 
    prompt = """Human: here is the data price:
        <text>""" + str(docs) + """</text>
        Question: """ + question + """ 
    \n\nAssistant: """
    return call_claude_sonet_stream(prompt)



# Preprocess the data
data = yf.download(ticker,start="2024-01-01", end=date.today().strftime("%Y-%m-%d"))
df = data.reset_index()[["Date", "Open", "High", "Low", "Close"]].copy()
df["Date"] = pd.to_datetime(df["Date"])
df["DayOfWeek"] = df["Date"].dt.dayofweek
df = df[(df["DayOfWeek"] != 5) & (df["DayOfWeek"] != 6)]
df = df.rename(columns={"Date": "ds", "Open": "open", "High": "high", "Low": "low", "Close": "y"})
            
# Train the Prophet model
m = Prophet(daily_seasonality=True, yearly_seasonality=True)
m.fit(df)
 # Make future dataframe and predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
forecast1 = forecast.rename(columns={'ds':'Date','yhat':'Predicted Prices','yhat_lower':'Predicted Lowest','yhat_upper':'Predicted Highest'})
forelist = ['Date', 'Predicted Prices', 'Predicted Lowest', 'Predicted Highest']
print(forecast)            
st.write("---")
st.subheader("Forecast Data")
st.write(forecast1[forelist])
st.write('---')
            
# Prediction interval
st.subheader("Prediction in Interval of Time")
Start = st.date_input('Enter start date', value=None)
End = st.date_input('Enter end date', value=None)
if Start != End:
    selected_forecast = forecast1.loc[(forecast1['Date'] > pd.to_datetime(Start)) & (forecast1['Date'] <= pd.to_datetime(End))]
    st.write(selected_forecast[forelist])
st.write("---")
            
# Forecasted Data Graphs
st.subheader('Forecasted Data Graphs')
st.write("Actual Prices v/s Predicted Prices")

# Create the candlestick chart
fig = go.Figure(data=[
    go.Candlestick(
        x=df['ds'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['y'],
        name='Actual Prices'
    ),
    go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red')
    )
])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig)
            
st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Suggestion
st.title('Dự đoán theo predict price')
st.write("---")
response = forecast_price(question="Dựa vào các chỉ số trên đưa ra phân tích giá chứng khoán trong thời gian tới,thời điểm cụ thể để mua vào và bán ra cổ phiếu, giá cổ phiếu là VND", docs = forecast)
st.write_stream(response)            
st.button("Exit")
st.write("---")