Portfolio Optimization - Efficient Frontier Analysis
A Python-based tool for portfolio optimization using Modern Portfolio Theory (MPT) and Mean-Variance Optimization to construct efficient portfolios from historical stock data.
Overview
This project implements portfolio optimization techniques to help investors find the optimal asset allocation that maximizes returns for a given level of risk. It visualizes the efficient frontier and identifies key portfolios including the maximum Sharpe ratio and minimum volatility portfolios.
Features

Historical Data Retrieval: Automatically downloads stock price data using Yahoo Finance API
Mean-Variance Optimization: Implements Markowitz portfolio theory to find optimal asset allocations
Monte Carlo Simulation: Generates 10,000 random portfolios to map the efficient frontier
Key Portfolio Identification:

Maximum Sharpe Ratio portfolio (best risk-adjusted returns)
Minimum Volatility portfolio (lowest risk)


Visualization: Creates an interactive efficient frontier plot showing:

All simulated portfolios colored by Sharpe ratio
Optimal portfolios highlighted
Individual asset positions


PyPortfolioOpt Integration: Validates results using the PyPortfolioOpt library


Prerequisites

Python 3.7 or higher
pip package manager

Required Libraries
pip install numpy pandas matplotlib yfinance pypfopt