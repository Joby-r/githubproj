import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
import plotly.graph_objects as go   


# apple = yf.Ticker("AAPL")

# hist = apple.history(period="1mo")

# print(hist)

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length" color="species")
fig.show()