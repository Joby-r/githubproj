import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
import plotly.graph_objects as go   
import datetime as dt
import random
import plotly.graph_objects as go

# CHECKING YFINANCE

# apple = yf.Ticker("AAPL")
# hist = apple.history(period="1mo")
# print(hist)

# CHECKING PLOTLY

# df = px.data.iris()
# fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
# fig.show()

# CHECKING SEABORN  

# stocks_df = pd.read_csv('stock-prices.csv')
# print(stocks_df.head())



stocks_df = pd.read_csv('stock-prices.csv')
fig = px.line(title= "AMZN scatter plot")
fig.add_scatter(x = stocks_df["Date"], y = stocks_df["AMZN"],name = "ADJ close")
fig.show() 

def plot_financial_data(df, title):
  fig=px.line(title=title)
  for i in df.columns[1:]:
    fig.add_scatter(x=df["Date"], y = df[i], name = i)
    fig.update_traces(line_width=5)
    fig.update_layout({"plot_bgcolor":"white"})
  fig.show()
plot_financial_data(stocks_df, "Test")


mercado_libre = yf.Ticker("MELI")
history = mercado_libre.history(start = "2014-01-02", end = "2022-12-17")
mercado_libre_dcp_df = history[['Close']].reset_index()
mercado_libre_dcp_df['Date'] = mercado_libre_dcp_df['Date'].dt.date
mercado_libre_dcp_df.rename(columns={"Close":"MELI"}, inplace = True)
mercado_libre_dcp_df


stocks_df["Date"] = pd.to_datetime(stocks_df["Date"])
mercado_libre_dcp_df["Date"] = pd.to_datetime(mercado_libre_dcp_df["Date"])

merged_df = pd.merge(stocks_df, mercado_libre_dcp_df, on="Date", how="inner")
merged_df

plot_financial_data(merged_df, "Test")

dr_merged_df = merged_df.iloc[:,1:].pct_change()*100
dr_merged_df.replace(np.nan,0,inplace=True)
dr_merged_df

dr_merged_df.insert(0, "Date", merged_df["Date"])
dr_merged_df

fig = px.histogram(dr_merged_df.drop(columns=["Date"]))
fig.update_layout({"plot_bgcolor":"white"})

plt.figure(figsize = (10,8))
sns.heatmap(dr_merged_df.drop(columns=["Date"]).corr(), annot=True)

def price_scaling_function(raw_prices_df):
  scaled_prices_df = raw_prices_df.copy()
  for i in raw_prices_df.columns[1:]:
    scaled_prices_df[i] = raw_prices_df[i]/raw_prices_df[i][0]
  return scaled_prices_df

scaled_df = price_scaling_function(merged_df)
scaled_df

plot_financial_data(scaled_df, "scaled DF plot test")

def generate_portfolio_weights(n):
  weights = []
  for i in range(n):
    weights.append(random.random())
  weights = weights/np.sum(weights)
  return weights

weights = generate_portfolio_weights(11)
print(weights)

initial_investment = 1000000
portfolio_df = merged_df.copy()
for i, stock in enumerate(scaled_df.columns[1:]):
  portfolio_df[stock] = weights[i]*scaled_df[stock]*initial_investment
portfolio_df.round(1)

def asset_allocation(df, weights, initial_investment):
  portfolio_df = df.copy()
  scaled_df = price_scaling_function(df)
  for i, stock in enumerate(scaled_df.columns[1:]):
    portfolio_df[stock] = scaled_df[stock]*weights[i]*initial_investment
  portfolio_df["Portfolio value (£)"] = portfolio_df[portfolio_df !="Date"].sum(axis = 1, numeric_only = True)
  portfolio_df["Daily Return (%)"] = portfolio_df["Portfolio value (£)"].pct_change(1)*100
  portfolio_df.replace(np.nan,0, inplace=True)
  return portfolio_df

n = len(merged_df.columns)-1

print("the No. stocks under consideration is {}".format(n))
weights = generate_portfolio_weights(n).round(6)
print("portfolio weights = {}".format(weights))

portfolio_df = asset_allocation(merged_df, weights, 1000000)
portfolio_df.round(2)

plot_financial_data(portfolio_df[["Date", "Daily Return (%)"]], "Portfolio Daily Return (%)")

plot_financial_data(portfolio_df.drop(["Portfolio value (£)", "Daily Return (%)"], axis = 1), 'Portfolio Positions')

plot_financial_data(portfolio_df[["Date", "Portfolio value (£)"]], "Portfolio Daily Return (£)")

def simulation_engine(weights, initial_investment):
    portfolio_df = asset_allocation(merged_df, weights, initial_investment)
    return_on_investment = ((portfolio_df["Portfolio value (£)"][-1:]-portfolio_df["Portfolio value (£)"][0])/portfolio_df["Portfolio value (£)"][0])*100
    portfolio_daily_return_df = portfolio_df.drop(columns=["Date","Portfolio value (£)", "Daily Return (%)"])
    portfolio_daily_return_df = portfolio_daily_return_df.pct_change(1)
    portfolio_expected_return = np.sum(weights*portfolio_daily_return_df.mean())*252
    covariance = portfolio_daily_return_df.cov()*252
    expected_volatility = np.sqrt(np.dot(weights.T,np.dot(covariance, weights)))
    rf = 0.05
    sharpe_ratio = (portfolio_expected_return-rf)/expected_volatility
    return portfolio_expected_return, expected_volatility, sharpe_ratio, portfolio_df["Portfolio value (£)"][-1:].values[0], return_on_investment.values[0]

initial_investment = 1000000
portfolio_metrics = simulation_engine(weights, initial_investment)

print("Expected portfolio return = {:.2f}%".format(portfolio_metrics[0]*100))
print("Portfolio SD = {:.2f}".format(portfolio_metrics[1]*100))
print("Sharpe Ratio = {:.2f}".format(portfolio_metrics[2]))
print("Portoflio final value = £{:.2f}".format(portfolio_metrics[3]))
print("ROI = {:.2f}".format(portfolio_metrics[4]))

portfolio_df.round(2)

sim_runs = 10000
initital_investment = 1000000
weights_runs = np.zeros((sim_runs,n))
sharpe_ratio_runs = np.zeros(sim_runs)
expected_portfolio_return_runs = np.zeros(sim_runs)
volatility_runs = np.zeros(sim_runs)
return_on_investment_runs = np.zeros(sim_runs)
final_value_runs = np.zeros(sim_runs)
for i in range(sim_runs):
  weights = generate_portfolio_weights(n)
  weights_runs[i,:] = weights
  expected_portfolio_return_runs[i], volatility_runs[i], sharpe_ratio_runs[i], final_value_runs[i], return_on_investment_runs[i] = simulation_engine(weights, initial_investment)
  print("simulation_runs = {}".format(i))
  print("weights = {}, final value = ${:.2f}, sharpe ratio = {:.2f}".format(weights_runs[i].round(3), final_value_runs[i], sharpe_ratio_runs[i]))
  print("\n")

sharpe_ratio_runs

print("The simualtion run that produced the highest sharpe ratio is: {}".format(sharpe_ratio_runs.argmax()))

sharpe_ratio_runs.max()
weights_runs

weights_runs[sharpe_ratio_runs.argmax(), :]

optimal_portfolio_return, optimal_volatility, optimal_sharpe_ratio, highest_final_value, optimal_return_on_investment = simulation_engine(weights_runs[sharpe_ratio_runs.argmax(),:], initial_investment)

print("Best portfolio metrics based on {} Monte Carlo simulations".format(sim_runs))
print("portfolio expected annual return = {:.2f}%".format(optimal_portfolio_return*100))
print("Portfolio SD = {:.2f}%".format(optimal_volatility*100))
print("Final Value = £{:.2f}".format(highest_final_value))
print("Return on investment = {:.2f}%".format(optimal_return_on_investment))
print("Optimal sharpe ratio = {:.2f}".format(optimal_sharpe_ratio))

sim_output_df = pd.DataFrame({"volatility":volatility_runs.tolist(), "portfolio_return":expected_portfolio_return_runs.tolist(), "Sharpe_ratio":sharpe_ratio_runs.tolist()})

fig = px.scatter(sim_output_df, x = "volatility", y = "portfolio_return", color = "Sharpe_ratio", size = "Sharpe_ratio", hover_data = ["Sharpe_ratio"])
fig.update_layout({"plot_bgcolor":"white"})
fig.show()

fig = px.scatter(sim_output_df, x = "volatility", y = "portfolio_return", color = "Sharpe_ratio", size = "Sharpe_ratio", hover_data = ["Sharpe_ratio"])
fig.add_trace(go.Scatter(x = [optimal_volatility], y = [optimal_portfolio_return], mode = "markers", name = "optimal_point", marker = dict(size = [40], color = "red")))
fig.update_layout(coloraxis_colorbar = dict(y = 0.7, dtick = 5))
fig.show()