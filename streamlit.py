# import streamlit as st
# import yfinance as yf
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import date

# # Title of the dashboard
# st.title('Monte Carlo Simulation of Stock Prices')

# # Sidebar options
# ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
# start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2016-01-01'))
# end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-06-01'))
# actual_end_date = st.sidebar.date_input('Actual data end date', value=pd.to_datetime('2024-01-28'))
# num_simulations = st.sidebar.number_input('Number of simulations', value=1000, min_value=100, max_value=10000)
# num_days = st.sidebar.number_input('Number of trading days', value=252, min_value=50, max_value=365)

# # Download stock data for simulation period
# @st.cache
# def load_data(ticker, start_date, end_date):
#     data = yf.download(ticker, start=start_date, end=end_date)
#     return data['Adj Close']

# prices = load_data(ticker, start_date, end_date)

# if not prices.empty:
#     # Calculate daily returns
#     returns = prices.pct_change().dropna()
#     log_returns = np.log(1 + returns)

#     # Set the last price and preallocate the simulation array
#     last_price = prices.iloc[-1]
#     simulations = np.zeros((num_days, num_simulations))

#     np.random.seed(0)  # for reproducibility
#     for i in range(num_simulations):
#         # Generate random daily log returns
#         random_log_returns = np.random.choice(log_returns, size=num_days)
#         # Sum the log returns to simulate the price path
#         cumulative_log_returns = np.cumsum(random_log_returns)
#         # Convert log returns to actual prices
#         simulations[:, i] = last_price * np.exp(cumulative_log_returns)

#     # Plot all simulations
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(simulations, alpha=0.1)  # lower opacity to see overlap
#     ax.set_title(f'Monte Carlo Simulation of {ticker}')
#     ax.set_xlabel('Days')
#     ax.set_ylabel('Price')
#     st.pyplot(fig)

#     # Calculate the median of the simulations
#     median_simulation = np.median(simulations, axis=1)

#     # Plot the simulations with the median path highlighted
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     ax2.plot(simulations, color='blue', alpha=0.05)  # Plot all simulations with low opacity
#     ax2.plot(median_simulation, color='red', label='Median Path')  # Plot the median path in red
#     ax2.set_title(f'Monte Carlo Simulation of {ticker} with Median Path Highlighted')
#     ax2.set_xlabel('Days')
#     ax2.set_ylabel('Price')
#     ax2.legend()
#     st.pyplot(fig2)

    
    
#     actual_data = yf.download(ticker, start=end_date, end=actual_end_date)
#     actual_prices = actual_data['Adj Close']

#     # Compute MSE between each simulation and actual stock prices
#     simulated_prices = simulations[:len(actual_prices), :]  # Consider only the part corresponding to actual data days
#     mse = np.mean((simulated_prices - actual_prices.values.reshape(-1, 1)) ** 2, axis=0)
#     min_mse_index = np.argmin(mse)  # Index of the simulation with the lowest MSE

#     # Plotting the best matching simulation against the actual data
#     fig3, ax3 = plt.subplots(figsize=(10, 6))
#     ax3.plot(actual_prices.index, actual_prices, label='Actual Data', color='blue')
#     full_simulation_index = pd.date_range(start=actual_prices.index[0], periods=252, freq='B')  # 'B' for business day frequency
#     ax3.plot(full_simulation_index, simulations[:, min_mse_index], label='Best Matching Full Simulation', color='red', alpha=0.5)
#     ax3.set_title('Comparison of Best Matching Monte Carlo Simulation with Actual Data')
#     ax3.set_xlabel('Date')
#     ax3.set_ylabel('Price')
#     ax3.legend()
#     st.pyplot(fig3)

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import plotly.graph_objs as go

# Title of the dashboard
st.title('Monte Carlo Simulation of Stock Prices')

# Sidebar options
ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2016-01-01'))
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-06-01'))
actual_end_date = st.sidebar.date_input('Actual data end date', value=pd.to_datetime('2024-01-28'))
num_simulations = st.sidebar.number_input('Number of simulations', value=1000, min_value=100, max_value=10000)
num_days = st.sidebar.number_input('Number of trading days', value=252, min_value=50, max_value=365)

# Download stock data for simulation period
@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

prices = load_data(ticker, start_date, end_date)

if not prices.empty:
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    log_returns = np.log(1 + returns)

    # Set the last price and preallocate the simulation array
    last_price = prices.iloc[-1]
    simulations = np.zeros((num_days, num_simulations))

    np.random.seed(0)  # for reproducibility
    for i in range(num_simulations):
        # Generate random daily log returns
        random_log_returns = np.random.choice(log_returns, size=num_days)
        # Sum the log returns to simulate the price path
        cumulative_log_returns = np.cumsum(random_log_returns)
        # Convert log returns to actual prices
        simulations[:, i] = last_price * np.exp(cumulative_log_returns)

    # Create a Plotly graph of all simulations
    traces = [go.Scatter(
        x=list(range(num_days)),
        y=simulations[:, i],
        mode='lines',
        line=dict(width=1),
        opacity=0.1
    ) for i in range(num_simulations)]
    layout = go.Layout(title=f'Monte Carlo Simulation of {ticker}', xaxis=dict(title='Days'), yaxis=dict(title='Price'))
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig)

    # Calculate the median of the simulations and plot with highlight
    median_simulation = np.median(simulations, axis=1)
    fig2 = go.Figure(data=traces)  # Use same traces for background simulations
    fig2.add_trace(go.Scatter(x=list(range(num_days)), y=median_simulation, mode='lines', line=dict(color='red', width=2), name='Median Path'))
    fig2.update_layout(title=f'Monte Carlo Simulation of {ticker} with Median Path Highlighted', xaxis=dict(title='Days'), yaxis=dict(title='Price'))
    st.plotly_chart(fig2)

    actual_data = yf.download(ticker, start=end_date, end=actual_end_date)
    actual_prices = actual_data['Adj Close']

    # Compute MSE between each simulation and actual stock prices
    simulated_prices = simulations[:len(actual_prices), :]
    mse = np.mean((simulated_prices - actual_prices.values.reshape(-1, 1)) ** 2, axis=0)
    min_mse_index = np.argmin(mse)  # Index of the simulation with the lowest MSE

    # Plotting the best matching simulation against the actual data
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices, mode='lines', name='Actual Data', line=dict(color='blue')))
    full_simulation_index = pd.date_range(start=actual_prices.index[0], periods=num_days, freq='B')  # 'B' for business day frequency
    fig3.add_trace(go.Scatter(x=full_simulation_index, y=simulations[:, min_mse_index], mode='lines', name='Best Matching Full Simulation', line=dict(color='red', width=2, dash='dash')))
    fig3.update_layout(title='Comparison of Best Matching Monte Carlo Simulation with Actual Data', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    st.plotly_chart(fig3)


