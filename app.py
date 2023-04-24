import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
import matplotlib.pyplot as plt
from matplotlib import style
from keras import models
import datetime

style.use('seaborn-v0_8-dark')
# start = '2010-01-01'
# end = '2022-12-23'

st.set_page_config(page_title='MarketProphet', page_icon='MarketProphet.png', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(':green[MarketProphet: Stock Market Prediction System]')
st.caption('Time Series Analysis Using stacked LSTM')
with st.sidebar:
    user_input = st.text_input('Enter Stock Ticker','AAPL')
    d1 = st.date_input("Enter the Lower Cutoff Date",datetime.date(2019, 1, 1))
    d2 = st.date_input("Enter the Higher Cutoff Date")
st.caption('Showing Data For:')
st.subheader(user_input)

yfin.pdr_override()

df = pdr.get_data_yahoo(user_input, start=d1, end=d2)

dfa = df.reset_index()
data_frame1 = dfa.head(5)
col1, col2 = st.columns(2)

col1.caption('Market Trends from Beginning')
col1.dataframe(data=data_frame1, width=None, height=None, use_container_width=False)
data_frame2 = dfa.tail(5)
col2.caption('Market Trends from Ending')
col2.dataframe(data=data_frame2, width=None, height=None, use_container_width=False)
# Describing Data

st.subheader(':red[Synopsis:] Technical Analysis of DataFrames')
st.write(df.describe())

# Visualizations.

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 8))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with _:blue[100 Days]_ Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 8))
plt.plot(df.Close)
plt.plot(ma100)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with _:blue[100 & 200 Days]_ Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 8))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# print(data_training.shape)
# print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Splitting data into x-train and y-train

# x_train = []
# y_train = []
#
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)

#Load my model.

model = models.load_model('keras_model1.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

#Predicting the plot
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
print(y_predicted)



scaler = scaler.scale_
print()
print(type(scaler))
print(scaler)

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Price vs Time chart against :red[Predictions] vs :blue[Original] Market Trend')

fig2 = plt.figure(figsize=(12, 8))
plt.plot(y_test, 'b', label='Original Price')
#plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig2)

fig2 = plt.figure(figsize=(12, 8))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig2)

fig2 = plt.figure(figsize=(12, 8))
#plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig2)
