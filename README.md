Writing a simple stock return predictor 

Pulling historical price data with yfinance, create features like moving averages, lagged returns and rolling volatility, then train a random forest (or LSTM) to predcit next day returns. Backtest a long/short strategy against buy and hold and evaluate it with a monte carlo resampling instead of a single equity curve.