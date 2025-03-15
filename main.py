import yfinance as yf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

INITIAL_BALANCE = 10000
RISK = 0.01

START_DATE = '2010-01-01'
END_DATE = '2025-03-10'


class stock_signals:
    def __init__(self, ticker, start_date, end_date):
        self.data = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date


    def get_data(self):
        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
        data['tp'] = (data['Close'] + data['High'] + data['Low']) / 3
        data['true_range'] = pd.concat([data['High'] - data['Low'], (data['High'] - data['Close'].shift(1)).abs(),(data['Low'] - data['Close'].shift(1)).abs()], axis=1).max(axis=1)
        data['ticker'] = f'{ticker}'

        data.columns = data.columns.rename({'Price': ''}) 
        data.columns = data.columns.droplevel(1)
        data.columns = [x.lower() for x in data.columns]

        self.data = data


    def calculate_technicals(self):
        self.data['sma_20'] = self.data['tp'].rolling(window=20).mean()
        self.data['std'] = self.data['tp'].rolling(window=20).std()

        self.data['upper_band'] = self.data['sma_20'] + 2.2 * self.data['std']
        self.data['lower_band'] = self.data['sma_20'] - 2.2 * self.data['std']

        delta = self.data['close'].diff(1)
        gains = delta.where(delta > 0,0)
        losses = -delta.where(delta < 0, 0)
        
        average_gains = gains.ewm(14, min_periods=1).mean()
        average_losses = losses.ewm(14, min_periods=1).mean()
        relative_strength = average_gains / average_losses

        self.data['rsi'] = 100 - (100 / (1 + relative_strength))

        self.data['atr'] = self.data['true_range'].rolling(14).mean()
        self.data['norm_atr'] = self.data['atr'] / self.data['close']

        self.data = self.data[19:]

        self.data = self.data[['close', 'sma_20', 'upper_band', 'lower_band', 'rsi', 'ticker', 'norm_atr']]

    
    def generate_signals(self):
        self.data['signal'] = ''

        for index,row in self.data.iterrows():

            if (row['close'] <= row['lower_band']) & (row['rsi'] < 40):
                self.data.loc[index, 'signal'] = 'buy'

            elif (row['close'] >= row['upper_band']) & (row['rsi'] > 60):
                self.data.loc[index, 'signal'] = 'sell'

            else:
                self.data.loc[index, 'signal'] = 'hold'


        signals = self.data['signal'].copy()
        last_signal = 'sell'
        new_signals = ['hold'] * len(signals)


        for i in range(len(signals)):

            if signals.iloc[i] == 'buy' and last_signal != 'buy':
                new_signals[i] = 'buy'
                last_signal = 'buy'

            elif signals.iloc[i] == 'sell' and last_signal != 'sell':
                new_signals[i] = 'sell'
                last_signal = 'sell'

        self.data['signal'] = new_signals


    def generate_plot(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(30,30) ,sharex=True)
        fig.suptitle(f'Plot of ${self.ticker} with Bollinger Bands and Macd')

        ax1.plot(self.data['close'], label='Price')
        ax1.plot(self.data['lower_band'], label='Lower bollinger band', color='green', linewidth=1)
        ax1.plot(self.data['upper_band'], label='Upper bollinger band', color='red', linewidth=1)

        ax2.plot(self.data['rsi'], label='Relative Strength Index', color='orange')

        buy_signals = self.data[self.data['signal'] == 'buy']
        sell_signals = self.data[self.data['signal'] == 'sell']
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=75, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=75, label='Sell')

        ax1.set_ylabel('Price')
        ax2.set_ylabel('RSI')
        plt.xlabel('Year', fontsize=12)

        fig.legend()
        plt.show()


class backtester:
    def __init__(self, initial_balance, data):
        self.data = data
        self.cash = initial_balance

        self.holdings = {}
        self.portfolio = {
            'Date' : [],
            'Cash' : [],
            'Portfolio Value' : []
        }


    def calculate_position_size(self, atr, price):
        position_size = (RISK * self.cash) / (atr * price)
        position_size = int(position_size)

        return position_size
    

    def backtest(self):

        last_date = None

        for i,row in self.data.iterrows():
            ticker = row['ticker']
            price = row['close']
            signal = row['signal']


            if signal == 'buy':
                position_size = self.calculate_position_size(row['norm_atr'], price)
                if self.cash >= position_size * price:
                    self.holdings[ticker] = self.holdings.get(ticker, 0) + position_size
                    self.cash -= position_size * price

            elif (signal == 'sell') and (self.holdings.get(ticker, 0) > 0):
                self.cash += self.holdings[ticker] * price
                self.holdings[ticker] = 0

            if last_date != i:
                portfolio_value = self.cash + sum(self.holdings.get(t, 0) * self.data[self.data['ticker'] == t].loc[i, 'close'] for t in self.data['ticker'].unique())
                self.portfolio['Date'].append(i)
                self.portfolio['Cash'].append(self.cash)
                self.portfolio['Portfolio Value'].append(portfolio_value)
                last_date = i

            print(i)

        self.portfolio = pd.DataFrame(self.portfolio)
    

    def output(self):
        print(self.portfolio)
        plt.plot(self.data.index.unique(), self.portfolio['Portfolio Value'], color='green', label='Strategy')
        plt.plot(self.data[self.data['ticker'] == 'SPY']['close'] * (10000 / self.data[self.data['ticker'] == 'SPY']['close'].iloc[0]), color='red', label='$SPY')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.title('Bollinger Bands and RSI Strategy plotted against $SPY')
        plt.legend()
        plt.show()

    
if __name__ == '__main__':
    tickers = ['AAPL', 'SPY', 'GLD', 'LLY', 'NVDA', 'WMT', 'MSFT']
    data = pd.DataFrame()
    
    for ticker in tickers:
        stock = stock_signals(ticker, START_DATE, END_DATE)
        stock.get_data()
        stock.calculate_technicals()
        stock.generate_signals()

        temp_data = stock.data[['ticker', 'signal', 'close', 'norm_atr']]
        data = pd.concat([data, temp_data])

    data = data.sort_index()

    portfolio = backtester(INITIAL_BALANCE, data)
    portfolio.backtest()
    portfolio.output()
