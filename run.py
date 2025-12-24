import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import warnings
from pypfopt import EfficientFrontier
warnings.filterwarnings('ignore')
tickers = ['AAPL', 'MSFT', 'GOOGL','AMZN','TSLA']
def get_data(tickers):
    expected_returns = np.zeros(len(tickers))
    returns_cov = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start="2020-01-01", end="2025-01-01",auto_adjust=False)
        prices = data['Adj Close']
        expected_returns[tickers.index(ticker)] = prices.pct_change().dropna().mean()
        monthly=prices.resample('MS').last()
        returns_cov = pd.concat([returns_cov,monthly.pct_change()],axis=1)
    annualized_expected_return = expected_returns* 252
    returns_cov = returns_cov.dropna().cov()*12
    return(returns_cov, annualized_expected_return)
    # print(data.head())
    # print(prices.head())
def optimize_portfolio(df,exp_returns):
    returns = exp_returns 
    cov_matrix =df.values
    num_portfolios = 10000
    results = np.zeros((4, num_portfolios))
    weights_record=[]
    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(len(exp_returns))
        weights /= np.sum(weights)  
        weights_record.append(weights)
        # Portfolio return
        portfolio_return = np.dot(weights, returns)
        
        # Portfolio risk (standard deviation)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate = 2%)
        sharpe = (portfolio_return - 0.02) / portfolio_std
        
        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = sharpe
        results[3,i]=i
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[0])
    
    # Get the actual weights for optimal portfolios
    max_sharpe_weights = weights_record[max_sharpe_idx]
    min_vol_weights = weights_record[min_vol_idx]
    
    return results, max_sharpe_weights, min_vol_weights

def plot_efficient_frontier(results, max_sharpe_weights, min_vol_weights, 
                            exp_returns, cov_matrix, tickers):
    """
    Visualize the efficient frontier and optimal portfolios.
    """
    plt.figure(figsize=(14, 9))
    
    # Scatter plot of all portfolios
    scatter = plt.scatter(results[0,:], results[1,:], 
                         c=results[2,:], cmap='viridis', 
                         alpha=0.5, s=10)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Max Sharpe portfolio
    max_sharpe_return = np.dot(max_sharpe_weights, exp_returns)
    max_sharpe_std = np.sqrt(np.dot(max_sharpe_weights.T, 
                                    np.dot(cov_matrix, max_sharpe_weights)))
    plt.scatter(max_sharpe_std, max_sharpe_return, 
               marker='*', color='red', s=700, 
               label=f'Max Sharpe Ratio ({max_sharpe_std:.2%}, {max_sharpe_return:.2%})',
               edgecolors='black', linewidths=2)
    
    # Min Volatility portfolio
    min_vol_return = np.dot(min_vol_weights, exp_returns)
    min_vol_std = np.sqrt(np.dot(min_vol_weights.T, 
                                 np.dot(cov_matrix, min_vol_weights)))
    plt.scatter(min_vol_std, min_vol_return, 
               marker='*', color='blue', s=700, 
               label=f'Min Volatility ({min_vol_std:.2%}, {min_vol_return:.2%})',
               edgecolors='black', linewidths=2)
    
    # Individual assets
    for i, ticker in enumerate(tickers):
        plt.scatter(np.sqrt(cov_matrix[i, i]), exp_returns[i],
                   marker='o', s=150, label=ticker, alpha=0.8)
    
    plt.xlabel('Volatility (Standard Deviation)', fontsize=13)
    plt.ylabel('Expected Annual Return', fontsize=13)
    plt.title('Efficient Frontier - Mean-Variance Optimization', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def optimize_portfolio_ef(df,exp_returns):
    ef = EfficientFrontier(exp_returns, df)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

df,exp_returns=get_data(tickers)
results, max_sharpe_weights, min_vol_weights=optimize_portfolio(df,exp_returns)
plot_efficient_frontier(results, max_sharpe_weights, min_vol_weights, exp_returns, df.values, tickers)
optimize_portfolio_ef(df,exp_returns)
# print(results)


