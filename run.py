import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
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