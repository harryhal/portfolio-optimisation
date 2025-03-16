# Portfolio Optimization using Markowitz Mean-Variance Model

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

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/portfolio-optimisation.git
cd portfolio-optimisation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the portfolio optimization with default settings:

```bash
python portfolio_optimization.py
```

This will optimize a portfolio consisting of AAPL, MSFT, AMZN, GOOGL, and META stocks using data from the last 3 years.

### Command-line Arguments

- `--tickers`: List of stock tickers (default: AAPL MSFT AMZN GOOGL META)
- `--start-date`: Start date in YYYY-MM-DD format (default: 3 years ago)
- `--end-date`: End date in YYYY-MM-DD format (default: today)
- `--risk-free-rate`: Risk-free rate (default: 0.0)
- `--target-return`: Target portfolio return (optional)
- `--target-risk`: Target portfolio risk/volatility (optional)

### Examples

Optimize a portfolio of technology and financial stocks:

```bash
python portfolio_optimization.py --tickers AAPL MSFT JPM BAC GS
```

Optimize a portfolio with a specific date range:

```bash
python portfolio_optimization.py --start-date 2018-01-01 --end-date 2023-01-01
```

Optimize a portfolio with a target annual return of 20%:

```bash
python portfolio_optimization.py --target-return 0.2
```

## Output

The script generates the following outputs:

1. Terminal output with portfolio statistics and optimal weights
2. Interactive HTML plots:
   - `efficient_frontier.html`: Efficient frontier with the optimal portfolio
   - `portfolio_weights.html`: Pie chart of optimal portfolio weights
3. Static PNG plots:
   - `efficient_frontier.png`: Efficient frontier with the optimal portfolio
   - `portfolio_weights.png`: Pie chart of optimal portfolio weights
4. CSV file:
   - `optimal_portfolio.csv`: Optimal portfolio weights

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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 