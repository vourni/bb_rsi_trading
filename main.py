import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

# Setting all import variables 
INITIAL_BALANCE = 100000
RISK = 0.1
STD_MULT = 2
UPPER_THRESH = 70
LOWER_THRESH = 30
START_DATE = '2015-01-01'
END_DATE = '2025-03-10'


class stock_signals:
    # Genertes signals
    def __init__(self, ticker, start_date, end_date):
        # Initializing
        self.data = None
        self.buy_price = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date


    def get_data(self):
        # Donwloading data
        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)

        # Creating new stats
        data['tp'] = (data['Close'] + data['High'] + data['Low']) / 3
        data['true_range'] = pd.concat([data['High'] - data['Low'], (data['High'] - data['Close'].shift(1)).abs(),(data['Low'] - data['Close'].shift(1)).abs()], axis=1).max(axis=1)
        data['ticker'] = f'{self.ticker}'

        # Adjusting columns
        data.columns = data.columns.rename({'Price': ''}) 
        data.columns = data.columns.droplevel(1)
        data.columns = [x.lower() for x in data.columns]

        # Saving
        self.data = data


    def calculate_technicals(self):
        # Calculating SMAs
        self.data['sma_20'] = self.data['tp'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['tp'].rolling(window=50).mean()
        self.data['sma_200'] = self.data['tp'].rolling(window=200).mean()

        # Calculating upper/lower bollinger bands
        self.data['std'] = self.data['tp'].rolling(window=20).std()
        self.data['upper_band'] = self.data['sma_20'] + STD_MULT * self.data['std']
        self.data['lower_band'] = self.data['sma_20'] - STD_MULT * self.data['std']

        # Calculating RSI
        delta = self.data['close'].diff()
        gains = delta.where(delta > 0,0)
        losses = -delta.where(delta < 0, 0)
        average_gains = gains.rolling(14, min_periods=1).mean()
        average_losses = losses.rolling(14, min_periods=1).mean()
        relative_strength = average_gains / average_losses
        self.data['rsi'] = 100 - (100 / (1 + relative_strength))
        # Calculating RSI percentile over 100 days
        self.data['rsi_oversold_percentile'] = self.data['rsi'].rolling(100).apply(lambda x: np.percentile(x, 20))
        self.data['rsi_overbought_percentile'] = self.data['rsi'].rolling(100).apply(lambda x: np.percentile(x, 80))

        # Calculating ATR
        self.data['atr'] = self.data['true_range'].rolling(50).mean()
        # Normalizing ATR
        self.data['norm_atr'] = (self.data['atr'] / self.data['close']) * 100
        # Finding ATR percentile
        self.data['atr_percentile'] = self.data['atr'].rolling(50).apply(lambda x: np.percentile(x, 75))

        # Calculating stop loss and take profit using atr percentiles
        self.data['stopl'] = self.data['close'] - self.data['atr'] * np.where(self.data['atr'] > self.data['atr_percentile'], 5, 2)
        self.data['takep'] = self.data['close'] + self.data['atr'] * np.where(self.data['atr'] > self.data['atr_percentile'], 20, 10)

        # Calculating ADX
        plus_dm = self.data['high'].diff()
        minus_dm = self.data['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / self.data['atr'])
        minus_di = 100 * (minus_dm.rolling(14).mean() / self.data['atr'])
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        self.data['adx'] = dx.rolling(14).mean()

        # Saving non NAN values
        self.data = self.data[100:]


    def generate_signals(self):
        # Generating signals
        action = None
        self.data['signal'] = ''

        for index,row in self.data.iterrows():
            # Iterating through rows
            # Useing indicators to generate signals
            if row['close'] <= row['lower_band'] and row['rsi'] < row['rsi_oversold_percentile'] and row['sma_50'] > row['sma_200'] and action != 'entry':
                # Setting signals
                self.data.loc[index, 'signal'] = 'entry'
                action = 'entry'
                # Setting exit points
                take_profit = row['takep']
                stop_loss = row['stopl']

            elif action == 'entry' and (stop_loss >= row['close'] or take_profit <= row['close']) and row['rsi'] > 55:
                # Setting signals
                self.data.loc[index, 'signal'] = 'exit'
                action = 'exit'
            
            else:
                # Setting hold to all other spaces that dont meet criteria 
                self.data.loc[index, 'signal'] = 'hold'


    def generate_plot(self):
        # Saving and plotting all data
        # Creating plots
        fig, (ax1, ax2) = plt.subplots(2, figsize=(30,30) ,sharex=True)
        fig.suptitle(f'Plot of ${self.ticker} with Bollinger Bands and Macd')

        # Plotting price and bands
        ax1.plot(self.data['close'], label='Price')
        ax1.plot(self.data['lower_band'], label='Lower bollinger band', color='green', linewidth=1)
        ax1.plot(self.data['upper_band'], label='Upper bollinger band', color='red', linewidth=1)
        ax1.plot(self.data['sma_50'], label='SMA 50', color='blue', linewidth=2)
        ax1.plot(self.data['sma_200'], label='SMA 200', color='orange', linewidth=2)

        # Plotting RSI
        ax2.plot(self.data['rsi'], label='Relative Strength Index', color='orange')

        # Plotting signals
        entry_signals = self.data[self.data['signal'] == 'entry']
        exit_signals = self.data[self.data['signal'] == 'exit']
        ax1.scatter(entry_signals.index, entry_signals['close'], marker='^', color='green', s=75, label='Entry')
        ax1.scatter(exit_signals.index, exit_signals['close'], marker='o', color='green', s=75, label='Exit')

        # Labeling
        ax1.set_ylabel('Price')
        ax2.set_ylabel('RSI')
        plt.xlabel('Year', fontsize=12)

        # Showing
        fig.legend()
        plt.show()



class backtester:
    def __init__(self, initial_balance, data):
        # Initializing
        self.data = data
        self.cash = initial_balance

        self.last_position = None
        self.buy_price = {}
        self.holdings = {}
        self.portfolio = {
            'Date' : [],
            'Position Value': [],
            'Cash' : [],
            'Portfolio Value' : []
        }
        self.trade_log = {
            'Date': [],
            'Ticker' : [],
            'Action': [],
            'Position' : [],
            'Price': [],
            'Stop Loss' : [],
            'Take Profit' : [],
            'Profit/Loss' : [],
        }


    def calculate_position_size(self, atr, price):
        # Calculating position size using ATR to adjust for volatility
        position_size = (RISK * self.cash) / (atr * price)
        # Set position size to integer
        position_size = int(position_size)
        # Making sure position isnt 0
        if position_size == 0:
            position_size = 1
        return position_size
    

    def backtest(self):
        # Backtest
        last_date = None

        # Itterating through rows of signal database
        for i,row in self.data.iterrows():
            # Setting quick variables
            ticker = row['ticker']
            price = row['close']
            signal = row['signal']

            if signal == 'entry':
                # Set position szie                
                position_size = self.calculate_position_size(row['atr'], price)

                # Check if cash is enough to cover estimated margin requirements
                if self.cash >= position_size * price:
                    # Save position size, buy price and reduce cash
                    self.holdings[ticker] = position_size
                    self.buy_price[ticker] = price
                    self.cash -= position_size * price


            elif signal == 'exit' and self.holdings.get(ticker, 0) > 0:
                # Return cash from trade
                self.cash += self.holdings.get(ticker, 0) * price

                # Reseting the holdings and saving the last position
                self.last_position = self.holdings.get(ticker, 0)
                self.holdings[ticker] = 0

            # Keep portfolio value only once per day
            if last_date != i:
                # Value of positions + cash
                value = sum(self.holdings.get(t, 0) * self.data[self.data['ticker'] == t].loc[i, 'close'] for t in list(self.holdings.keys()) if self.holdings.get(t, 0) > 0)
                portfolio_value = self.cash + value

                # Appending portfolio
                self.portfolio['Date'].append(i)
                self.portfolio['Position Value'].append(value)
                self.portfolio['Cash'].append(self.cash)
                self.portfolio['Portfolio Value'].append(portfolio_value)

                # Resetting last date
                last_date = i

            # Append trade log dictionary
            if signal in ['entry', 'exit']:
                self.trade_log['Date'].append(i)
                self.trade_log['Ticker'].append(ticker)
                self.trade_log['Position'].append(self.last_position)
                self.trade_log['Action'].append(signal)
                self.trade_log['Price'].append(price)
                self.trade_log['Stop Loss'].append(row['stopl'] if signal == 'entry' else np.nan)
                self.trade_log['Take Profit'].append(row['takep'] if signal == 'entry' else np.nan)
                self.trade_log['Profit/Loss'].append(
                    (price - self.buy_price.get(ticker, 0)) * self.last_position if signal == 'exit' else np.nan
                )

            # Keep track of progress
            print(i)


        # Set dicitonaries to dataframes
        self.portfolio = pd.DataFrame(self.portfolio)
        self.trade_log = pd.DataFrame(self.trade_log)
    

    # Print portoflio, trade log and graph
    def output(self):
        self.portfolio['Returns'] = self.portfolio['Portfolio Value'].diff(1)
        sharpe_ratio = self.portfolio['Returns'].mean() / self.portfolio['Returns'].std()

        stats = {
            'Sharpe Ratio' : float(sharpe_ratio),
            'Total Returns': float((self.portfolio['Portfolio Value'].iloc[-1] / INITIAL_BALANCE - 1) * 100),
            'Annualized Return' : (self.portfolio["Portfolio Value"].iloc[-1] / INITIAL_BALANCE) ** (1/15) - 1,
        }

        print(pd.DataFrame([stats]))

        plt.plot(self.data.index.unique(), self.portfolio['Portfolio Value'], color='green', label='Strategy')
        plt.plot(self.data[self.data['ticker'] == 'SPY']['close'] * (INITIAL_BALANCE / self.data[self.data['ticker'] == 'SPY']['close'].iloc[0]), color='red', label='$SPY')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.title('Bollinger Bands and RSI Strategy plotted against $SPY')
        plt.legend()
        plt.show()


    
# Run script
if __name__ == '__main__':
    # Initializing
    tickers = ['SPY', 'AAPL', 'GLD', 'LLY', 'NVDA', 'WMT', 'MSFT', 'GOVT', 'TEAM', 'FXI', 'IWM', 'AVGO', 'JPM', 'JNJ', 'GOOG', 'CVNA', 'MNST', 'TSCO', 'DECK', 'TPL', 'FICO', 'NVR', 'AMD', 'CELH', 'SMCI', 'GME', 'AMC', 'VRT']
    data = pd.DataFrame()
    
    # Looping through tickers
    for ticker in tickers:
        # Generating data for each stock
        stock = stock_signals(ticker, START_DATE, END_DATE)
        stock.get_data()
        stock.calculate_technicals()
        stock.generate_signals()

        data = pd.concat([data, stock.data])

    # Sorting signal dataframe
    data = data.sort_index()

    # Print portfolio
    portfolio = backtester(INITIAL_BALANCE, data)
    portfolio.backtest()
    portfolio.output()

