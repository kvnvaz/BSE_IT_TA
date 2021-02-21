import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('seaborn')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
#  Basic Techincal Indicators for BSE Software Stocks 
#### *A Cannabiz venture, cause with us your returns might not be high. But you always are...*
This app gives visualizations of basic technical indicators
""")

st.sidebar.header('Select Stock and Time Range')

def user_input_features():
    stockcode = st.sidebar.selectbox('Select Stock Code',('TCS.NS', 'INFY.BO', 'WIPRO.BO', 'HCLTECH.BO', 'LTI.BO', 'TECHM.BO', 'MPHASIS.BO', 'MINDTREE.NS', 'PERSISTENT.NS', 'OFSS.NS', 'TATAELXSI.BO','ULTRACEMCO.NS','ULTRACEMCO.BO'))
    st.sidebar.write('You selected:', stockcode)
    interval = st.sidebar.selectbox('Select Time Interval',('1d', '1wk', '1mo'))
    st.sidebar.write('You selected:', interval)
    data = {'stockcode': stockcode, 'interval' : interval}
    features= pd.DataFrame(data, index=[0])
    return features

df= user_input_features()

st.subheader("Displaying Charts For Stock " )
st.write(df)
#Define the ticker symbol
TickSym= df['stockcode'][0]
interval= df['interval'][0]
today=datetime.today().strftime('%Y-%m-%d')
#Import Ticker Data
TickData= yf.Ticker(TickSym)

#Get Historical prices for this ticker
if(interval=='1mo'): 
    stockdata = TickData.history(interval=interval,start='1999-1-1',end=today)
elif(df['interval'][0]=='1wk'):
    stockdata = TickData.history(interval=interval,start='2009-1-1',end=today)
else:
    stockdata = TickData.history(interval=interval,start='2019-1-1',end=today)
#st.write("""Closing Price Chart""")
#st.line_chart(Tickdf.Close)
#st.write("""Volume Chart""")
#st.bar_chart(Tickdf.Volume)


plt.figure(figsize=(60,18))
plt.plot(stockdata.index, stockdata['Close'], label = 'Adjusted Close Price')
plt.title('Adj. Close Price History')
plt.xlabel(" January 1990 - Current", fontsize=18)
plt.ylabel(" Price in INR ",fontsize =18 )
plt.show()

"""### Technical Indicator Functions"""

#Create functions to calculate Simple Moving Average SMA
def SMA(data, period=30, column=('Close')):
  return data[column].rolling(window=period).mean()

#Create functions to calculate Exponential Moving Average SMA
def EMA(data, period=20, column=('Close')):
  return data[column].ewm(span=period, adjust=False).mean()

#Create function to calculate the moving average convergence/divergence (MACD)
def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
  #calcluate Short term Exponential Moving Average
  ShortEMA = EMA(data,period_short,column)
  #calcluate long term Exponential Moving AverageDa
  LongEMA = EMA(data,period_long,column)
  #calculate the moving average convergence/divergence (MACD)
  data['MACD'] = ShortEMA - LongEMA
  #calculate the signal line
  data['Signal_Line'] = EMA(data, period_signal, column ='MACD')
  return data

def RSI(data, period = 14, column ='CLose'):
  ##############################################################
  #Data Pre-processing to calculate Relative Strength Index(RSI)
  ##############################################################
  #Get difference in price from the previous day
  delta = data['Close'].diff(1)

  #Null Handling
  delta = delta.dropna()

  #Seperate Difference into up and down indicators
  up = delta.copy()
  down = delta.copy()
  up[up<0]=0
  down[down>0]=0

  ##############################################################
  # Period is usually taken as 14 days (can be changed)
  ##############################################################
  #Calculate Avg. Gain and Avg. Loss
  AvgGain = up.rolling(window=period).mean()
  AvgLoss = abs(down.rolling(window=period).mean())
  ##############################################################
  # Relative Strength Index(RSI) Calculation
  ##############################################################
  #Calculate Relative Strength
  RS= AvgGain / AvgLoss
  #Calculate Relative Strength Index
  RSI= 100.0 - (100.0 / (1.0 + RS))
  data['RSI'] = RSI

  return data

##############################################################
# Calling Function
##############################################################

MACD(stockdata)
RSI(stockdata)

#Assigning Value to columns
stockdata[('SMA')]=SMA(stockdata)
stockdata[('EMA')]=EMA(stockdata)
#stockdata=stockdata.set_index(pd.DatetimeIndex(stockdata['Date'].values))

""" Close price and trend comparion. SMA signifies long term trend and EMA highlights recent trend
"""

##############################################################
 #Plot SMA Chart
 ##############################################################

column_list=['Close','SMA','EMA']
st.line_chart(stockdata[column_list])
#plt.title("Close Price Plot")
#plt.ylabel(" Price in INR")
#plt.xlabel("Date")

""" Stock Volume Bar Chart. This helps understand demand and supply volume.
"""
column_list=['Volume']
st.bar_chart(stockdata[column_list])

""" Relative Strength Index. Value above 70 indicates that the stock is over-purchased while value below 30 suggests over-selling
"""
column_list=['RSI']
st.line_chart(stockdata[column_list])
#plt.title("RSI Plot")
#plt.ylabel(" Price in INR")
#plt.xlabel("Date")
#plt.axhline(70,color='green')
#plt.axhline(30,color='red')
#plt.show()
#st.pyplot()

""" Moving Average Convergence Divergence. """ 
"""
Reffered to as bull (MACD) line and bear (Signal_Line) line. If MACD line is above the Signal_Line it is a sign of bullish market and vice versa
"""
column_list=['MACD','Signal_Line']
st.line_chart(stockdata[column_list])
#plt.title("MACD Plot")
#plt.ylabel(" Price in INR")
#plt.xlabel("Date")
#plt.show()

#st.pyplot()