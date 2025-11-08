import numpy as np
import matplotlib.pyplot as plt


returns = np.array([0.12, 0.08, 0.05])  
cov_matrix = np.array([
    [0.04, 0.006, 0.002],  
    [0.006, 0.0225, 0.003],
    [0.002, 0.003, 0.01]
])

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

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier - Mean-Variance Optimization')
plt.show()
print(weights)