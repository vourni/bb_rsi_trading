import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

INITIAL_BALANCE = 100000
RISK = 0.1

STD_MULT = 2
UPPER_THRESH = 70
LOWER_THRESH = 30

START_DATE = '2015-01-01'
END_DATE = '2025-03-10'


class stock_signals:
    def __init__(self, ticker, start_date, end_date):
        self.data = None
        self.buy_price = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date



    def get_data(self):
        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
        data['tp'] = (data['Close'] + data['High'] + data['Low']) / 3
        data['true_range'] = pd.concat([data['High'] - data['Low'], (data['High'] - data['Close'].shift(1)).abs(),(data['Low'] - data['Close'].shift(1)).abs()], axis=1).max(axis=1)
        data['ticker'] = f'{self.ticker}'

        data.columns = data.columns.rename({'Price': ''}) 
        data.columns = data.columns.droplevel(1)
        data.columns = [x.lower() for x in data.columns]

        self.data = data



    def calculate_technicals(self):
        self.data['sma_20'] = self.data['tp'].rolling(window=20).mean()
        self.data['sma_50'] = self.data['tp'].rolling(window=50).mean()
        self.data['sma_200'] = self.data['tp'].rolling(window=200).mean()
        self.data['std'] = self.data['tp'].rolling(window=20).std()
        self.data['upper_band'] = self.data['sma_20'] + (STD_MULT+0.5) * self.data['std']
        self.data['lower_band'] = self.data['sma_20'] - STD_MULT * self.data['std']

        delta = self.data['close'].diff()
        gains = delta.where(delta > 0,0)
        losses = -delta.where(delta < 0, 0)
        average_gains = gains.rolling(14, min_periods=1).mean()
        average_losses = losses.rolling(14, min_periods=1).mean()
        relative_strength = average_gains / average_losses
        self.data['rsi'] = 100 - (100 / (1 + relative_strength))
        self.data['rsi_oversold_percentile'] = self.data['rsi'].rolling(100).apply(lambda x: np.percentile(x, 20))
        self.data['rsi_overbought_percentile'] = self.data['rsi'].rolling(100).apply(lambda x: np.percentile(x, 80))

        self.data['atr'] = self.data['true_range'].rolling(50).mean()
        self.data['norm_atr'] = (self.data['atr'] / self.data['close']) * 100
        self.data['atr_percentile'] = self.data['atr'].rolling(50).apply(lambda x: np.percentile(x, 75))
        self.data['stopl'] = self.data['close'] - self.data['norm_atr'] * np.where(self.data['atr'] > self.data['atr_percentile'], 5, 2)
        self.data['takep'] = self.data['close'] + self.data['norm_atr'] * np.where(self.data['atr'] > self.data['atr_percentile'], 20, 10)

        plus_dm = self.data['high'].diff()
        minus_dm = self.data['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / self.data['atr'])
        minus_di = 100 * (minus_dm.rolling(14).mean() / self.data['atr'])
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        self.data['adx'] = dx.rolling(14).mean()

        self.data = self.data[100:]

    

    def generate_signals(self):
        long_action = None
        short_action = None
        self.data['signal'] = ''

        for index,row in self.data.iterrows():

            if row['close'] <= row['lower_band'] and row['rsi'] < row['rsi_oversold_percentile'] and row['sma_50'] > row['sma_200'] and long_action != 'entry':
                self.data.loc[index, 'signal'] = 'long_entry'
                long_action = 'entry'
                self.buy_price = row['close']
                long_take_profit = row['takep']
                long_stop_loss = row['stopl']

            elif row['close'] >= row['upper_band'] and row['rsi'] > row['rsi_overbought_percentile'] and row['sma_50'] < row['sma_200'] and short_action != 'entry':
                self.data.loc[index, 'signal'] = 'short_entry'
                short_action = 'entry'
                self.buy_price = row['close']
                short_take_profit = row['takep']
                short_stop_loss = row['stopl']

            elif long_action == 'entry' and (long_stop_loss >= row['close'] or long_take_profit <= row['close']) and row['rsi'] > 55:
                self.data.loc[index, 'signal'] = 'long_exit'
                long_action = 'exit'

            elif short_action == 'entry' and (short_take_profit >= row['close'] or short_stop_loss <= row['close']) and row['rsi'] < 45:
                self.data.loc[index, 'signal'] = 'short_exit'
                short_action = 'exit'
            
            else:
                self.data.loc[index, 'signal'] = 'hold'



    def generate_plot(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(30,30) ,sharex=True)
        fig.suptitle(f'Plot of ${self.ticker} with Bollinger Bands and Macd')

        ax1.plot(self.data['close'], label='Price')
        ax1.plot(self.data['lower_band'], label='Lower bollinger band', color='green', linewidth=1)
        ax1.plot(self.data['upper_band'], label='Upper bollinger band', color='red', linewidth=1)

        ax2.plot(self.data['rsi'], label='Relative Strength Index', color='orange')

        long_entry_signals = self.data[self.data['signal'] == 'long_entry']
        short_entry_signals = self.data[self.data['signal'] == 'short_entry']
        long_exit_signals = self.data[self.data['signal'] == 'long_exit']
        short_exit_signals = self.data[self.data['signal'] == 'short_exit']
        ax1.scatter(long_entry_signals.index, long_entry_signals['close'], marker='^', color='green', s=75, label='Long Entry')
        ax1.scatter(short_entry_signals.index, short_entry_signals['close'], marker='v', color='red', s=75, label='Short Entry')
        ax1.scatter(long_exit_signals.index, long_exit_signals['close'], marker='o', color='green', s=75, label='Long Exit')
        ax1.scatter(short_exit_signals.index, short_exit_signals['close'], marker='o', color='red', s=75, label='Short Exit')

        ax1.set_ylabel('Price')
        ax2.set_ylabel('RSI')
        plt.xlabel('Year', fontsize=12)

        fig.legend()
        plt.show()



class backtester:
    def __init__(self, initial_balance, data):
        self.data = data
        self.cash = initial_balance

        self.short_buy_price = {}
        self.long_buy_price = {}
        self.holdings = {}
        self.portfolio = {
            'Date' : [],
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
        #    'Profit/Loss' : [],
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


            if signal == 'long_entry':
                position_size = self.calculate_position_size(row['atr'], price)

                if self.cash >= position_size * price:
                    self.holdings[ticker] = self.holdings.get(ticker, 0) + position_size
                    self.long_buy_price[ticker] = price
                    self.cash -= position_size * price


            if signal == 'short_entry':
                position_size = self.calculate_position_size(row['atr'], price)

                if self.cash >= position_size * price:
                    self.holdings[ticker] = self.holdings.get(ticker, 0) - position_size
                    self.short_buy_price[ticker] = price


            elif signal == 'long_exit' and self.holdings.get(ticker, 0) > 0:
                self.cash += self.holdings[ticker] * price
                self.holdings[ticker] = 0

            elif signal == 'short_exit' and self.holdings.get(ticker, 0) < 0:
                self.cash += abs(self.holdings[ticker]) * (self.short_buy_price[ticker] - price)
                self.holdings[ticker] = 0

            if last_date != i:
                long_value = sum(self.holdings.get(t, 0) * self.data[self.data['ticker'] == t].loc[i, 'close'] for t in list(self.holdings.keys()) if self.holdings.get(t) > 0)
                short_value = sum(abs(self.holdings.get(t, 0)) * (self.short_buy_price[t] - self.data[self.data['ticker'] == t].loc[i, 'close']) for t in list(self.holdings.keys()) if self.holdings.get(t) < 0)
                portfolio_value = self.cash + long_value + short_value

                self.portfolio['Date'].append(i)
                self.portfolio['Cash'].append(self.cash)
                self.portfolio['Portfolio Value'].append(portfolio_value)

                last_date = i

            if signal in ['long_entry', 'long_exit', 'short_entry', 'short_exit']:
                self.trade_log['Date'].append(i)
                self.trade_log['Ticker'].append(ticker)
                self.trade_log['Position'].append(position_size if signal in ['long_entry', 'short_entry'] else np.nan)
                self.trade_log['Action'].append(signal)
                self.trade_log['Price'].append(price)
                self.trade_log['Stop Loss'].append(row['stopl'] if signal in ['long_entry', 'short_entry'] else np.nan)
                self.trade_log['Take Profit'].append(row['takep'] if signal in ['long_entry', 'short_entry'] else np.nan)
                #self.trade_log['Profit/Loss'].append(((price - self.long_buy_price[ticker]) * position_size if signal == 'long_exit' else 
                #                                      (self.short_buy_price[ticker] - price) * position_size) if signal in ['long_exit', 'short_exit'] else np.nan)

            print(i)


        self.portfolio = pd.DataFrame(self.portfolio)
        self.trade_log = pd.DataFrame(self.trade_log)
    


    def output(self):
        print(self.portfolio)
        print(self.trade_log)
        plt.plot(self.data.index.unique(), self.portfolio['Portfolio Value'], color='green', label='Strategy')
        plt.plot(self.data[self.data['ticker'] == 'SPY']['close'] * (INITIAL_BALANCE / self.data[self.data['ticker'] == 'SPY']['close'].iloc[0]), color='red', label='$SPY')
        plt.xlabel('Date')
        plt.ylabel('USD')
        plt.title('Bollinger Bands and RSI Strategy plotted against $SPY')
        plt.legend()
        plt.show()

    

if __name__ == '__main__':
    tickers = ['SPY', 'AAPL', 'GLD', 'LLY', 'NVDA', 'WMT', 'MSFT', 'GOVT', 'TEAM', 'FXI', 'IWM', 'AVGO', 'JPM', 'JNJ', 'GOOG', 'CVNA', 'MNST', 'TSCO', 'DECK', 'TPL', 'FICO', 'NVR', 'AMD', 'CELH', 'SMCI', 'GME', 'AMC', 'VRT']
    data = pd.DataFrame()
    
    for ticker in tickers:
        stock = stock_signals(ticker, START_DATE, END_DATE)
        stock.get_data()
        stock.calculate_technicals()
        stock.generate_signals()
        #stock.generate_plot()

        data = pd.concat([data, stock.data])

    data = data.sort_index()

    #print(data[data['signal'].isin(['long_entry', 'long_exit', 'short_entry', 'short_exit'])][['close', 'stopl', 'takep', 'atr', 'norm_atr', 'signal', 'adx']])

    portfolio = backtester(INITIAL_BALANCE, data)
    portfolio.backtest()
    portfolio.output()
