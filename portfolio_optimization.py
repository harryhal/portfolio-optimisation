#!/usr/bin/env python3
"""
Portfolio Optimization using Markowitz Mean-Variance Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
import argparse

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame: Historical adjusted close prices for the specified stocks
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    # Set auto_adjust=True to get adjusted close prices directly
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # In newer versions of yfinance, with auto_adjust=True, we need to use 'Close' instead of 'Adj Close'
    if 'Close' in data.columns:
        data = data['Close']
    elif 'Adj Close' in data.columns:
        data = data['Adj Close']
    else:
        # If neither column exists, try to determine the structure
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, the first level is the column name, second level is the ticker
            if 'Close' in data.columns.levels[0]:
                data = data['Close']
            elif 'Adj Close' in data.columns.levels[0]:
                data = data['Adj Close']
    
    # If only one ticker is provided, yfinance returns a Series instead of DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=tickers)
    
    # Check for missing data
    missing_data = data.isna().sum()
    if missing_data.any():
        print("Warning: Missing data detected:")
        print(missing_data[missing_data > 0])
        
        # Fill missing values with forward fill method
        data = data.fillna(method='ffill')
        
    return data

def calculate_returns(prices):
    """
    Calculate daily returns from price data
    
    Args:
        prices (DataFrame): Historical price data
        
    Returns:
        DataFrame: Daily returns
    """
    return prices.pct_change().dropna()

def calculate_portfolio_performance(weights, returns):
    """
    Calculate portfolio performance metrics
    
    Args:
        weights (array): Portfolio weights
        returns (DataFrame): Historical returns
        
    Returns:
        tuple: (expected return, volatility, Sharpe ratio)
    """
    # Convert to numpy arrays for faster computation
    returns_array = returns.values
    
    # Calculate expected returns (annualized)
    expected_return = np.sum(returns.mean() * weights) * 252
    
    # Calculate portfolio volatility (annualized)
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = expected_return / portfolio_volatility
    
    return expected_return, portfolio_volatility, sharpe_ratio

def negative_sharpe_ratio(weights, returns):
    """
    Calculate negative Sharpe ratio (for minimization)
    
    Args:
        weights (array): Portfolio weights
        returns (DataFrame): Historical returns
        
    Returns:
        float: Negative Sharpe ratio
    """
    return -calculate_portfolio_performance(weights, returns)[2]

def optimize_portfolio(returns, risk_free_rate=0.0, target_return=None, target_risk=None, max_weight=None):
    """
    Optimize portfolio weights using Markowitz Mean-Variance Model
    
    Args:
        returns (DataFrame): Historical returns
        risk_free_rate (float): Risk-free rate (default: 0.0)
        target_return (float, optional): Target portfolio return
        target_risk (float, optional): Target portfolio risk
        max_weight (float, optional): Maximum weight for any single asset
        
    Returns:
        dict: Optimized portfolio information
    """
    num_assets = len(returns.columns)
    args = (returns,)
    
    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    
    # If max_weight is specified, update bounds
    if max_weight is not None:
        bounds = tuple((0, min(1, max_weight)) for _ in range(num_assets))
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # If target return is specified, add constraint
    if target_return is not None:
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: calculate_portfolio_performance(x, returns)[0] - target_return
        })
    
    # If target risk is specified, add constraint
    if target_risk is not None:
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: calculate_portfolio_performance(x, returns)[1] - target_risk
        })
    
    # Optimize for maximum Sharpe ratio
    optimization_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimized weights
    optimal_weights = optimization_result['x']
    
    # Calculate portfolio performance with optimized weights
    expected_return, volatility, sharpe_ratio = calculate_portfolio_performance(optimal_weights, returns)
    
    # Create result dictionary
    result = {
        'weights': dict(zip(returns.columns, optimal_weights)),
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio
    }
    
    return result

def generate_efficient_frontier(returns, num_portfolios=100):
    """
    Generate the efficient frontier
    
    Args:
        returns (DataFrame): Historical returns
        num_portfolios (int): Number of portfolios to generate
        
    Returns:
        DataFrame: Portfolio metrics for the efficient frontier
    """
    num_assets = len(returns.columns)
    results = []
    
    # Calculate minimum volatility portfolio
    min_vol_result = minimize(
        lambda weights: calculate_portfolio_performance(weights, returns)[1],
        np.array([1/num_assets] * num_assets),
        bounds=tuple((0, 1) for _ in range(num_assets)),
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    min_vol_weights = min_vol_result['x']
    min_vol_return, min_vol_volatility, _ = calculate_portfolio_performance(min_vol_weights, returns)
    
    # Calculate maximum return portfolio
    max_return_weights = np.zeros(num_assets)
    max_return_idx = returns.mean().argmax()
    max_return_weights[max_return_idx] = 1
    max_return, max_return_volatility, _ = calculate_portfolio_performance(max_return_weights, returns)
    
    # Generate portfolios along the efficient frontier
    target_returns = np.linspace(min_vol_return, max_return, num_portfolios)
    
    for target_return in target_returns:
        result = optimize_portfolio(returns, target_return=target_return)
        results.append({
            'return': result['expected_return'],
            'volatility': result['volatility'],
            'sharpe_ratio': result['sharpe_ratio'],
            'weights': result['weights']
        })
    
    return pd.DataFrame(results)

def plot_efficient_frontier(efficient_frontier, optimal_portfolio, risk_free_rate=0.0):
    """
    Plot the efficient frontier with the optimal portfolio
    
    Args:
        efficient_frontier (DataFrame): Efficient frontier data
        optimal_portfolio (dict): Optimal portfolio data
        risk_free_rate (float): Risk-free rate
    """
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add efficient frontier
    fig.add_trace(go.Scatter(
        x=efficient_frontier['volatility'],
        y=efficient_frontier['return'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2)
    ))
    
    # Add optimal portfolio
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio['volatility']],
        y=[optimal_portfolio['expected_return']],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    # Add capital market line if risk-free rate is provided
    if risk_free_rate is not None:
        # Calculate the slope of the capital market line (Sharpe ratio)
        x_range = np.linspace(0, max(efficient_frontier['volatility']) * 1.2, 100)
        y_values = risk_free_rate + optimal_portfolio['sharpe_ratio'] * x_range
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_values,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier and Optimal Portfolio',
        xaxis_title='Volatility (Standard Deviation)',
        yaxis_title='Expected Return',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )
    
    # Save the plot as HTML
    fig.write_html('efficient_frontier.html')
    print("Efficient frontier plot saved as 'efficient_frontier.html'")
    
    # Also create a static matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(efficient_frontier['volatility'], efficient_frontier['return'], 'b-', linewidth=2, label='Efficient Frontier')
    plt.scatter(optimal_portfolio['volatility'], optimal_portfolio['expected_return'], color='red', marker='*', s=200, label='Optimal Portfolio')
    
    if risk_free_rate is not None:
        x_range = np.linspace(0, max(efficient_frontier['volatility']) * 1.2, 100)
        y_values = risk_free_rate + optimal_portfolio['sharpe_ratio'] * x_range
        plt.plot(x_range, y_values, 'g--', linewidth=2, label='Capital Market Line')
    
    plt.title('Efficient Frontier and Optimal Portfolio')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('efficient_frontier.png', dpi=300)
    print("Efficient frontier plot saved as 'efficient_frontier.png'")

def plot_portfolio_weights(weights, title='Optimal Portfolio Weights'):
    """
    Plot portfolio weights as a pie chart
    
    Args:
        weights (dict): Portfolio weights
        title (str): Plot title
    """
    # Sort weights by value
    sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    
    # Create pie chart with Plotly
    fig = go.Figure(data=[go.Pie(
        labels=list(sorted_weights.keys()),
        values=list(sorted_weights.values()),
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=plt.cm.tab20.colors)
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_white'
    )
    
    # Save the plot as HTML
    fig.write_html('portfolio_weights.html')
    print("Portfolio weights plot saved as 'portfolio_weights.html'")
    
    # Also create a static matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.pie(
        list(sorted_weights.values()),
        labels=list(sorted_weights.keys()),
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab20.colors
    )
    plt.axis('equal')
    plt.title(title)
    plt.savefig('portfolio_weights.png', dpi=300)
    print("Portfolio weights plot saved as 'portfolio_weights.png'")

def main():
    """Main function to run the portfolio optimization"""
    parser = argparse.ArgumentParser(description='Portfolio Optimization using Markowitz Mean-Variance Model')
    
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
                        help='List of stock tickers (default: AAPL MSFT AMZN GOOGL META)')
    
    # Fix the date range issue - use today as end date and 3 years back as start date
    today = datetime.now()
    three_years_ago = today - timedelta(days=365*3)
    
    parser.add_argument('--start-date', type=str, default=three_years_ago.strftime('%Y-%m-%d'),
                        help='Start date in YYYY-MM-DD format (default: 3 years ago)')
    
    parser.add_argument('--end-date', type=str, default=today.strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format (default: today)')
    
    parser.add_argument('--risk-free-rate', type=float, default=0.0,
                        help='Risk-free rate (default: 0.0)')
    
    parser.add_argument('--target-return', type=float, default=None,
                        help='Target portfolio return (optional)')
    
    parser.add_argument('--target-risk', type=float, default=None,
                        help='Target portfolio risk/volatility (optional)')
    
    parser.add_argument('--max-weight', type=float, default=None,
                        help='Maximum weight for any single asset (optional, e.g., 0.3 for 30%)')
    
    args = parser.parse_args()
    
    # Fetch stock data
    stock_prices = fetch_stock_data(args.tickers, args.start_date, args.end_date)
    
    # Calculate returns
    returns = calculate_returns(stock_prices)
    
    # Print basic statistics
    print("\nBasic Statistics (Annualized):")
    print("Expected Returns:")
    print((returns.mean() * 252).to_string())
    print("\nVolatility (Standard Deviation):")
    print((returns.std() * np.sqrt(252)).to_string())
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(returns.corr().round(2).to_string())
    
    # Optimize portfolio
    optimal_portfolio = optimize_portfolio(
        returns, 
        risk_free_rate=args.risk_free_rate,
        target_return=args.target_return,
        target_risk=args.target_risk,
        max_weight=args.max_weight
    )
    
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
    plot_efficient_frontier(efficient_frontier, optimal_portfolio, args.risk_free_rate)
    plot_portfolio_weights(optimal_portfolio['weights'])
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Ticker': list(optimal_portfolio['weights'].keys()),
        'Weight': list(optimal_portfolio['weights'].values())
    })
    results_df = results_df.sort_values('Weight', ascending=False)
    results_df.to_csv('optimal_portfolio.csv', index=False)
    print("Optimal portfolio weights saved to 'optimal_portfolio.csv'")

if __name__ == "__main__":
    main() 