import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3, json
import pandas_datareader as pdr
import base

st.set_page_config(page_title="Phân tích kỹ thuật cổ phiếu", page_icon="img/favicon.ico", layout="wide")
st.title('Phân tích kỹ thuật cổ phiếu')


snp500 = pd.read_csv("SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()    

ticker = st.selectbox('Chọn mã chứng khoán', symbols)

# price
def get_stock_price(ticker, history=500):
    today = dt.datetime.today()
    start_date = today - dt.timedelta(days=history)
    df_price = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    #print(df_price)
    return df_price

def plot_stock_price(ticker, history=500):
    df_price = get_stock_price(ticker, history)
    # Create the price chart
    fig = go.Figure(df_price=[go.Scatter(x=df['Date'], y=df_price['Adj Close'], mode='lines', name='Adjusted Close')])
    fig.update_layout(
        title=f'Stock Price for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis_tickprefix='$'
    )
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

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

    modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
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


def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
        df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df

def calcBollinger(data, size):
        df = data.copy()
        df["sma"] = df['Adj Close'].rolling(size).mean()
        df["bolu"] = df["sma"] + df['Adj Close'].rolling(size).std(ddof=0) 
        df["bold"] = df["sma"] - df['Adj Close'].rolling(size).std(ddof=0) 
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

def analyse_stock():
    st.subheader('Moving Average Convergence Divergence (MACD)')
    numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 
    startMACD = dt.datetime.today()-dt.timedelta(numYearMACD * 365)
    endMACD = dt.datetime.today()
    dataMACD = yf.download(ticker,startMACD,endMACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()
   
    figMACD = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)
    # print(df_macd)
    figMACD.add_trace(
        go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['Adj Close'],
                    name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema12'],
                    name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['ema26'],
                    name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
                ),
            row=1, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['macd'],
                    name = "MACD Line"
                ),
            row=2, col=1
        )
    
    figMACD.add_trace(
            go.Scatter(
                    x = df_macd['Date'],
                    y = df_macd['signal'],
                    name = "Signal Line"
                ),
            row=2, col=1
        )
    
    figMACD.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figMACD.update_yaxes(tickprefix="$")
    st.plotly_chart(figMACD, use_container_width=True)

     # Forecast stock
    st.title('Dự đoán theo chỉ số kỹ thuật')
    st.write("---")
    st.subheader('Dự đoán với chỉ số MACD')
    response = forecast_price(question="Dựa vào các chỉ số trên đưa ra phân tích giá chứng khoán trong thời gian tới,thời điểm, đưa ra giá mua vào và bán ra cổ phiếu cụ thể, giá cổ phiếu là VND", docs = df_macd)
    st.write_stream(response)
    
    st.subheader('Bollinger Band')
    coBoll1, coBoll2 = st.columns(2)
    with coBoll1:
        numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
        
    with coBoll2:
        windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
    
    startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
    endBoll = dt.datetime.today()
    dataBoll = yf.download(ticker,startBoll,endBoll)
    df_boll = calcBollinger(dataBoll, windowSizeBoll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
            go.Scatter(
                    x = df_boll['Date'],
                    y = df_boll['bolu'],
                    name = "Upper Band"
                )
        )
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['sma'],
                        name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                    )
            )
    
    
    figBoll.add_trace(
                go.Scatter(
                        x = df_boll['Date'],
                        y = df_boll['bold'],
                        name = "Lower Band"
                    )
            )
    
    figBoll.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="left",
        x=0
    ))
    
    figBoll.update_yaxes(tickprefix="$")
    st.plotly_chart(figBoll, use_container_width=True)
    
    st.write("---")
    st.subheader('Dự đoán với chỉ số BOLL')
    response = forecast_price(question="Dựa vào các chỉ số trên phân tích giá chứng khoán trong thời gian tới,thời điểm, đưa ra giá mua vào và bán ra cổ phiếu cụ thể, giá cổ phiếu là VND", docs = df_boll)
    st.write_stream(response)

if st.button("Do Analysis"):
    analyse_stock()