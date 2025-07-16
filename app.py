# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('apple_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

data = load_data()

# Sidebar
st.sidebar.title("📈 Stock Forecasting")
model_choice = st.sidebar.selectbox("Select Forecasting Model", ['LSTM', 'ARIMA', 'SARIMA', 'Prophet'])

# Display raw data
st.title("📊 Apple Stock Forecasting App")
st.write("Data Preview:")
st.line_chart(data['Close'])

# Forecasting Logic
if model_choice == 'LSTM':
    st.subheader("🔮 LSTM Prediction")

    # Load pre-trained model and scaler
    model = load_model('models/lstm_model.h5')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Prepare data
    close_data = data[['Close']]
    scaled_data = scaler.transform(close_data)
    
    time_step = 60
    x_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred = model.predict(x_input)
    forecast = scaler.inverse_transform(pred)[0][0]

    st.success(f"📅 Forecast for next day: ${forecast:.2f}")

elif model_choice == 'ARIMA':
    st.subheader("🔮 ARIMA Forecast (30 Days)")
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax, label='Actual')
    forecast.index = pd.date_range(data.index[-1], periods=30, freq='B')
    forecast.plot(ax=ax, label='Forecast', color='red')
    plt.legend()
    st.pyplot(fig)

elif model_choice == 'SARIMA':
    st.subheader("🔮 SARIMA Forecast (30 Days)")
    model = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=30)
    
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax, label='Actual')
    forecast.predicted_mean.plot(ax=ax, label='Forecast', color='green')
    plt.fill_between(forecast.conf_int().index,
                     forecast.conf_int().iloc[:, 0],
                     forecast.conf_int().iloc[:, 1], color='k', alpha=0.1)
    plt.legend()
    st.pyplot(fig)

elif model_choice == 'Prophet':
    st.subheader("🔮 Prophet Forecast (365 Days)")

    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    fig = model.plot(forecast)
    st.pyplot(fig)
