import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
tickers = ['AAPL', 'MSFT', 'GOOGL']
def get_data(tickers):
    expected_returns = np.zeros(len(tickers))
    returns_cov = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start="2020-01-01", end="2025-01-01",auto_adjust=False)
        prices = data['Adj Close']
        expected_returns[tickers.index(ticker)] = prices.pct_change().dropna().mean()
        monthly=prices.resample('MS').first()
        returns_cov = pd.concat([returns_cov,monthly.pct_change()],axis=1)
    annualized_expected_return = (1 + expected_returns) ** 252 - 1
    returns_cov = returns_cov.dropna().cov()
    return(returns_cov, annualized_expected_return)
    # print(data.head())
    # print(prices.head())
def optimize_portfolio(df,exp_returns):
    returns = exp_returns 
    cov_matrix =df.values
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(3)
        weights /= np.sum(weights)  # Normalize to sum to 1
        
        # Portfolio return
        portfolio_return = np.dot(weights, returns)
        
        # Portfolio risk (standard deviation)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate = 2%)
        sharpe = (portfolio_return - 0.02) / portfolio_std
        
        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = sharpe
    print(weights)
df,exp_returns=get_data(tickers)
optimize_portfolio(df,exp_returns)

# optimize_portfolio(df,exp_returns)
# plt.figure(figsize=(10, 6))
# plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
# plt.colorbar(label='Sharpe Ratio')
# plt.xlabel('Risk (Standard Deviation)')
# plt.ylabel('Expected Return')
# plt.title('Efficient Frontier - Mean-Variance Optimization')
# plt.show()
