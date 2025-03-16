#!/usr/bin/env python3
"""
Example script for portfolio optimization
"""

import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization import (
    fetch_stock_data,
    calculate_returns,
    optimize_portfolio,
    generate_efficient_frontier,
    plot_efficient_frontier,
    plot_portfolio_weights
)

def main():
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    risk_free_rate = 0.02  # 2% risk-free rate
    
    print(f"Running portfolio optimization for {tickers}")
    print(f"Time period: {start_date} to {end_date}")
    print(f"Risk-free rate: {risk_free_rate:.2%}")
    
    # Fetch stock data
    stock_prices = fetch_stock_data(tickers, start_date, end_date)
    
    # Calculate returns
    returns = calculate_returns(stock_prices)
    
    # Print basic statistics
    print("\nBasic Statistics (Annualized):")
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * (252 ** 0.5)
    
    # Create a DataFrame for better display
    stats_df = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': annual_returns / annual_volatility
    })
    
    print(stats_df.sort_values('Sharpe Ratio', ascending=False).round(4))
    
    # Optimize portfolio
    optimal_portfolio = optimize_portfolio(returns, risk_free_rate=risk_free_rate)
    
    # Print results
    print("\nOptimal Portfolio:")
    print(f"Expected Annual Return: {optimal_portfolio['expected_return']:.4f} ({optimal_portfolio['expected_return']*100:.2f}%)")
    print(f"Annual Volatility: {optimal_portfolio['volatility']:.4f} ({optimal_portfolio['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
    
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in sorted(optimal_portfolio['weights'].items(), key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Generate efficient frontier
    efficient_frontier = generate_efficient_frontier(returns)
    
    # Plot results
    plot_efficient_frontier(efficient_frontier, optimal_portfolio, risk_free_rate)
    plot_portfolio_weights(optimal_portfolio['weights'])
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Ticker': list(optimal_portfolio['weights'].keys()),
        'Weight': list(optimal_portfolio['weights'].values())
    })
    results_df = results_df.sort_values('Weight', ascending=False)
    results_df.to_csv('example_optimal_portfolio.csv', index=False)
    print("Optimal portfolio weights saved to 'example_optimal_portfolio.csv'")
    
    # Compare with equal-weight portfolio
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    print("\nEqual-Weight Portfolio for Comparison:")
    for ticker, weight in equal_weights.items():
        print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Plot equal-weight portfolio for comparison
    plot_portfolio_weights(equal_weights, title='Equal-Weight Portfolio')

if __name__ == "__main__":
    main() 