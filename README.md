# Portfolio Optimisation using Markowitz Mean-Variance Model

This project implements a portfolio optimization algorithm based on the Markowitz Mean-Variance Model. It fetches historical stock price data from Yahoo Finance and calculates the optimal portfolio weights to maximize the Sharpe ratio.

## Features

- Fetch historical stock price data from Yahoo Finance
- Calculate optimal portfolio weights using the Markowitz Mean-Variance Model
- Generate the efficient frontier
- Visualize results with interactive Plotly charts and static Matplotlib plots
- Support for custom constraints (target return, target risk)
- Command-line interface for easy usage

## Requirements

- Python 3.6+
- Required packages: numpy, pandas, matplotlib, scipy, yfinance, plotly

## Theory

### Markowitz Mean-Variance Model

The Markowitz Mean-Variance Model, developed by Harry Markowitz in 1952, is a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk. The model assumes that investors are risk-averse, meaning that given two portfolios that offer the same expected return, investors will prefer the less risky one.

The key insights of the model are:
1. The risk of a portfolio is not the weighted average of individual asset risks, but depends on how the assets move together (their covariance).
2. By combining assets with different patterns of returns, investors can reduce portfolio risk without sacrificing expected return.

### Efficient Frontier

The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk.

### Sharpe Ratio

The Sharpe ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. It is calculated as:

Sharpe Ratio = (Expected Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation

A higher Sharpe ratio indicates a more attractive risk-adjusted return.
