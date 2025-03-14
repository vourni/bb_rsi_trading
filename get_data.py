import yfinance as yf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


class stock_signals:
    
    def __init__(self, ticker, start_date, end_date):
        self.data = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date


    def get_data(self):

        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
        data['TP'] = (data['Close'] + data['High'] + data['Low']) / 3

        data.columns = data.columns.rename({'Price': f'{self.ticker}'}) 
        data.columns = data.columns.droplevel(1)

        self.data = data


    def calculate_technicals(self):

        self.data['SMA_20'] = self.data['TP'].rolling(window=20).mean()
        self.data['std'] = self.data['TP'].rolling(window=20).std()

        self.data['upper_band'] = self.data['SMA_20'] + 2 * self.data['std']
        self.data['lower_band'] = self.data['SMA_20'] - 2 * self.data['std']

        delta = self.data['Close'].diff(1)
        gains = delta.where(delta > 0,0)
        losses = -delta.where(delta < 0, 0)
        
        average_gains = gains.ewm(14, min_periods=1).mean()
        average_losses = losses.ewm(14, min_periods=1).mean()
        relative_strength = average_gains / average_losses

        self.data['RSI'] = 100 - (100 / (1 + relative_strength))

        self.data = self.data[19:]

        self.data = self.data[['Close', 'SMA_20', 'upper_band', 'lower_band', 'RSI']]

    
    def generate_signals(self):

        self.data['signal'] = ''

        for index,row in self.data.iterrows():
            if (row['Close'] <= row['lower_band']) & (row['RSI'] < 30):
                self.data.loc[index, 'signal'] = 'buy'
            elif (row['Close'] >= row['upper_band']) & (row['RSI'] > 70):
                self.data.loc[index, 'signal'] = 'sell'
            else:
                self.data.loc[index, 'signal'] = 'hold'

    def alternate_signals(self):
        
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

        ax1.plot(self.data['Close'], label='Price')
        ax1.plot(self.data['lower_band'], label='Lower bollinger band', color='green', linewidth=1)
        ax1.plot(self.data['upper_band'], label='Upper bollinger band', color='red', linewidth=1)

        ax2.plot(self.data['RSI'], label='Relative Strength Index', color='orange')

        buy_signals = self.data[self.data['signal'] == 'buy']
        sell_signals = self.data[self.data['signal'] == 'sell']
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=75, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=75, label='Sell')

        ax1.set_ylabel('Price')
        ax2.set_ylabel('RSI')
        plt.xlabel('Year', fontsize=12)

        fig.legend()
        plt.show()

    
    def kelly_criterion(self):

        returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()


if __name__ == '__main__':
    stocks = ['AAPL', 'SPY']
    start_date = '2010-01-01'
    end_date = '2025-03-10'



    for stock in stocks:
        stock = stock_signals(stock, start_date, end_date)
        stock.get_data()
        stock.calculate_technicals()
        stock.generate_signals()
        stock.alternate_signals()
        #stock.generate_plot()
        print(stock.data[(stock.data['signal'] == 'buy') | (stock.data['signal'] == 'sell')])

