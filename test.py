            #elif row['close'] >= row['upper_band'] and row['rsi'] > row['rsi_overbought_percentile'] and row['sma_50'] < row['sma_200'] and short_action != 'entry':
            #    self.data.loc[index, 'signal'] = 'short_entry'
            #    short_action = 'entry'
            #    self.buy_price = row['close']
            #   short_take_profit = row['takep']
            #   short_stop_loss = row['stopl']


            #elif short_action == 'entry' and (short_take_profit >= row['close'] or short_stop_loss <= row['close']) and row['rsi'] < 45:
            #    self.data.loc[index, 'signal'] = 'short_exit'
            #    short_action = 'exit'