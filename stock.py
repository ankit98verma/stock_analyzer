from strargparser import StrArgParser
import nsepy as nse
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle as pk

_alpha = 0.2


class Stock:

    def __init__(self, name, tracker, sector="Null"):
        self.name = name
        self.tracker = tracker
        self.iter_value = -1
        self.current_price = -1
        self.hist_data = None
        self.sector = sector
        self.get_current_price()

        self.close = self.tracker + '_Close'
        self.low = self.tracker + '_Low'
        self.high = self.tracker + '_High'
        self.open = self.tracker + '_Open'

    def get_hist_data(self, start_time, end_time):
        if type(start_time) is str:
            s = datetime.strptime(start_time, '%Y-%m-%d')
        else:
            s = start_time
        if type(end_time) is str:
            e = datetime.strptime(end_time, '%Y-%m-%d')
        else:
            e = end_time
        tmp = nse.get_history(symbol=self.tracker, start=s, end=e)
        tmp.index = pd.to_datetime(tmp.index)
        c_new = []
        for k in tmp.columns:
            c_new.append(self.tracker + "_" + k)
        tmp.columns = c_new
        return tmp

    def get_current_price(self):
        current_quote = nse.get_quote(self.tracker)
        self.current_price = current_quote['lastPrice']
        return self.current_price

    def get_tracker(self):
        return self.tracker

    def get_name(self):
        return self.name

    def set_name(self, n):
        self.name = n

    def set_tracker(self, n):
        self.tracker = n
        self.get_current_price()

    def __repr__(self):
        return 'Stock(Name = ' + self.name + ', Tracker = ' + self.tracker + '): Current Price = ' + \
               str(self.current_price)

    def __bool__(self):
        return self.tracker == ''

    def __iter__(self):
        if self.iter_value < 0:
            i = 0
            while i < self.iter_value:
                i += 1
                yield self.get_current_price()
        else:
            while 1:
                yield self.get_current_price()

    def fill_hist_data(self, start, end):
        self.hist_data = self.get_hist_data(start, end)

    def get_clt(self, parameter):
        returns = np.log(self.hist_data[parameter] / self.hist_data[parameter].shift(1))
        r = returns.cumsum().apply(np.exp)
        r = pd.DataFrame(r)
        return r

    def get_data(self, parameter):
        r = pd.DataFrame(self.hist_data[parameter])
        r.columns = [self.tracker + "_" + parameter]
        return r

    def get_rolling_data(self, parameter, window_sizes, win_type=None, func=np.mean, func_name='mean'):
        parameter = self.tracker + "_" + parameter
        k = pd.DataFrame(self.hist_data[parameter])
        for w in window_sizes:
            k[func_name + "_" + str(w)] = k[parameter].rolling(window=w, win_type=win_type).apply(func, raw=False)
        return k

    def get_bollinger_bonds_indicator(self, n, x):
        mb = self.get_rolling_data('Close', [n])
        std = self.get_rolling_data('Close', [n], func=np.std, func_name='std')
        mb['up_' + str(n)] = mb['mean_' + str(n)] + x * std['std_' + str(n)]
        mb['down_' + str(n)] = mb['mean_' + str(n)] - x * std['std_' + str(n)]
        k1 = np.where(mb[self.tracker + "_Close"] >= mb['up_' + str(n)], -1, 0)
        k2 = np.where(mb[self.tracker + "_Close"] <= mb['down_' + str(n)], 1, 0)
        mb['positions'] = k1 + k2
        return mb, ['-', 'b--', 'g--', 'r--']

    @staticmethod
    def plot_bollinger_bonds(bolli_data, n):
        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()

        columns = bolli_data.columns
        columns = columns[:-1]
        bolli_data[columns].plot(ax=ax2)

        ax2.fill_between(bolli_data.index, bolli_data['up_' + str(n)], bolli_data['down_' + str(n)], alpha=_alpha)
        bolli_data['positions'].plot(style='r', ax=ax)

        ax.grid()

    def get_moving_average_indicator(self, win1, win2):
        parameter = 'Close'
        roll_data = self.get_rolling_data(parameter, [win1, win2])
        if win1 < win2:
            roll_data['positions'] = np.where(roll_data['mean_' + str(win1)] > roll_data['mean_' + str(win2)], 1, -1)
        else:
            roll_data['positions'] = np.where(roll_data['mean_' + str(win1)] > roll_data['mean_' + str(win2)], -1, 1)
        return roll_data, ['-', 'b--', 'g--']

    @staticmethod
    def plot_moving_average(mov_data, win1, win2):
        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()

        columns = mov_data.columns
        columns = columns[:-1]
        mov_data[columns].plot(ax=ax2)
        ax2.fill_between(mov_data.index, mov_data['mean_' + str(win1)], mov_data['mean_' + str(win2)],
                         where=mov_data['positions'] >= 1, facecolor='green', alpha=_alpha)
        ax2.fill_between(mov_data.index, mov_data['mean_' + str(win1)], mov_data['mean_' + str(win2)],
                         where=mov_data['positions'] < 1, facecolor='red', alpha=_alpha)
        mov_data['positions'].plot(ax=ax)
        ax.grid()

    def get_ichimoku_kinko_hyo_indicator(self):
        parameter = 'Close'
        res = self.get_rolling_data(parameter, [9], func=np.max, func_name='highs')
        lows = self.get_rolling_data(parameter, [9], func=np.min, func_name='lows')
        res['Tenkan_sen'] = (res['highs_9'] + lows['lows_9']) / 2
        res = res.drop('highs_9', axis=1)

        highs = self.get_rolling_data(parameter, [26], func=np.max, func_name='highs')
        lows = self.get_rolling_data(parameter, [26], func=np.min, func_name='lows')
        res['Kijun_sen'] = (highs['highs_26'] + lows['lows_26']) / 2

        res['Chikou_Span'] = res[self.tracker + '_Close'].shift(-26)

        res['Senkou_Span_A'] = ((res['Kijun_sen'] + res['Tenkan_sen']) / 2).shift(26)

        highs = self.get_rolling_data(parameter, [52], func=np.max, func_name='highs')
        lows = self.get_rolling_data(parameter, [52], func=np.min, func_name='lows')
        res['Senkou_Span_B'] = ((highs['highs_52'] + lows['lows_52']) / 2).shift(26)

        # TODO: Add position data
        return res, ['b-', 'r--', 'b--', 'g--', 'r-', 'g-']

    @staticmethod
    def plot_ichimoku_kinko_hyo(p_data, style):
        ax = p_data.plot(style=style)
        ax.fill_between(p_data.index, p_data['Senkou_Span_A'], p_data['Senkou_Span_B'],
                        where=p_data['Senkou_Span_A'] >= p_data['Senkou_Span_B'],
                        facecolor='red', alpha=_alpha)
        ax.fill_between(p_data.index, p_data['Senkou_Span_A'], p_data['Senkou_Span_B'],
                        where=p_data['Senkou_Span_A'] < p_data['Senkou_Span_B'],
                        facecolor='green', alpha=_alpha)
        ax.yaxis.tick_right()
        ax.grid()

    def get_rsi_indicator(self):
        res = pd.DataFrame(self.hist_data[self.close])
        p = res[self.close].pct_change()
        gains = p.mask(p < 0, 0)
        gains.fillna(0, inplace=True)
        avg_gains = gains.rolling(14).mean()

        losses = p.mask(p > 0, 0)
        losses.fillna(0, inplace=True)
        losses = losses * -1
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        res['RSI'] = 100 - 100 / (1 + rs)

        # TODO: Add indicator also
        return res, ['-', 'r--']

    @staticmethod
    def plot_rsi(p_data):
        fig, ax = plt.subplots(2, 1)

        p_data[p_data.columns[0]].plot(ax=ax[0])
        ax[0].set_ylabel('Close value')
        ax[0].yaxis.tick_right()
        ax[0].grid()

        p_data['RSI'].plot(style='r--', ax=ax[1])

        high_mark = 70 * np.ones(p_data.index.size)
        low_mark = 30 * np.ones(p_data.index.size)
        mid_mark = 50 * np.ones(p_data.index.size)
        ax[1].plot(p_data.index, high_mark)
        ax[1].plot(p_data.index, low_mark)
        ax[1].plot(p_data.index, mid_mark)
        ax[1].fill_between(p_data.index, high_mark, mid_mark, alpha=_alpha, facecolor='green')
        ax[1].fill_between(p_data.index, mid_mark, low_mark, alpha=_alpha, facecolor='red')
        ax[1].set_ylabel('RSI')

    def get_macd_indicator(self):
        res = pd.DataFrame(self.hist_data[self.close])
        ema_12 = res[self.close].ewm(com=12 - 1).mean()
        ema_26 = res[self.close].ewm(com=26 - 1).mean()

        res['macd'] = ema_12 - ema_26
        res['signal_line'] = res['macd'].ewm(span=9).mean()
        res['histogram'] = res['macd'] - res['signal_line']

        return res, ['-', 'b--', 'r--', 'g--']

    @staticmethod
    def plot_macd(p_data):
        fig, ax = plt.subplots(2, 1)

        ax2 = ax[0].twinx()

        p_data[p_data.columns[0]].plot(ax=ax2)
        ax2.set_ylabel('Close value')
        ax2.grid()
        ax2.legend()

        p_data['macd'].plot(ax=ax[1])
        p_data['signal_line'].plot(ax=ax[1])

        ax[0].bar(p_data.index, p_data['histogram'], label='MACD histogram', color='red', alpha=_alpha)
        ax[0].legend()
        ax[1].legend()

    def get_parabolic_sar_indicator(self):
        af_const = 0.04
        af_max_const = 0.2

        af = af_const
        res = pd.DataFrame(self.hist_data[self.close])
        sar = np.zeros(res.size)

        sar[0] = (self.hist_data[self.high][0] + self.hist_data[self.low][0]) / 2
        if self.hist_data[self.close][0] < self.hist_data[self.close][1]:
            # Upward trend
            trend = True
            ep = self.hist_data[self.low][0]
        else:
            # downward trend
            trend = False
            ep = self.hist_data[self.high][0]

        for i in range(1, len(sar)):
            if trend and sar[i - 1] >= self.hist_data[self.low][i] or \
                    (not trend and sar[i - 1] <= self.hist_data[self.high][i]):
                trend = not trend
                sar[i] = ep
                af = af_const
                ep = self.hist_data[self.high][i] if trend else self.hist_data[self.low][i]
            else:
                if trend:
                    if self.hist_data[self.high][i] > ep:
                        ep = self.hist_data[self.high][i]
                        af = min(af + af_const, af_max_const)
                else:
                    if self.hist_data[self.low][i] < ep:
                        ep = self.hist_data[self.low][i]
                        af = min(af + af_const, af_max_const)
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

                if trend:
                    if sar[i] > self.hist_data[self.low][i] or sar[i] > self.hist_data[self.low][i - 1]:
                        sar[i] = min(self.hist_data[self.low][i], self.hist_data[self.low][i - 1])
                else:
                    if sar[i] < self.hist_data[self.high][i] or sar[i] < self.hist_data[self.high][i - 1]:
                        sar[i] = min(self.hist_data[self.high][i], self.hist_data[self.high][i - 1])

        res = pd.DataFrame(self.hist_data[self.close])
        res['SAR'] = pd.DataFrame(sar, index=res.index)

        # TODO: Add positions to it

        return res, ['-', '.']

    @staticmethod
    def plot_sar(p_data, style):
        ax = p_data.plot(style=style)
        ax.yaxis.tick_right()
        ax.grid()
        plt.title('SAR plot')

        plt.legend()

    def get_stochastic_indicator(self):

        res = pd.DataFrame(self.hist_data[self.close])
        l14 = self.get_rolling_data(self.close, [14], func=np.min, func_name='low')
        h14 = self.get_rolling_data(self.close, [14], func=np.max, func_name='high')

        res['K'] = (res[self.close] - l14['low_14']) / (h14['high_14'] - l14['low_14']) * 100
        res['D_fast'] = res['K'].rolling(3).mean()
        res['D_slow'] = res['D_fast'].rolling(3).mean()

        # TODO: add positions data
        return res, ['-', 'b--', 'r--', 'g--']

    @staticmethod
    def plot_stochastic(p_data):
        fig, ax = plt.subplots(2, 1)

        p_data[p_data.columns[0]].plot(ax=ax[0])
        ax[0].set_ylabel('Close value')
        ax[0].yaxis.tick_right()
        ax[0].grid()

        p_data['K'].plot(style='r--', ax=ax[1])
        p_data['D_fast'].plot(style='b--', ax=ax[1])
        ax[1].yaxis.tick_right()

        high_mark = 80 * np.ones(p_data.index.size)
        low_mark = 20 * np.ones(p_data.index.size)
        mid_mark = 50 * np.ones(p_data.index.size)
        ax[1].plot(p_data.index, high_mark)
        ax[1].plot(p_data.index, low_mark)
        ax[1].plot(p_data.index, mid_mark)
        ax[1].fill_between(p_data.index, high_mark, mid_mark, alpha=_alpha, facecolor='green')
        ax[1].fill_between(p_data.index, mid_mark, low_mark, alpha=_alpha, facecolor='red')
        ax[1].set_ylabel('Stochastic indicator')
        ax[1].legend()

    def get_adx_indicator(self):
        high = self.hist_data[self.high]
        low = self.hist_data[self.low]
        res = pd.DataFrame(self.hist_data[self.close])

        res['upmoves'] = high - high.shift(1)
        res['lowmoves'] = low.shift(1) - low
        res['isup'] = (res['upmoves'] > res['lowmoves']) & (res['upmoves'] > 0)
        res['+dm'] = res['upmoves'].mask(~res['isup'], 0)

        res['isdown'] = (res['upmoves'] < res['lowmoves']) & (res['lowmoves'] > 0)
        res['-dm'] = res['lowmoves'].mask(~res['isdown'], 0)

        res.drop(['isup', 'isdown'], axis=1, inplace=True)

        tmp = pd.DataFrame(res[self.close].shift(1))
        tmp['h'] = high
        tmp['low'] = low
        tmp['l_s'] = low.shift(1)

        res['tr'] = tmp[['h', 'l_s']].max(axis=1) - tmp[['low', self.close]].min(axis=1)
        res['atr'] = res['tr'].ewm(com=14 - 1).mean()
        res['+di'] = 100 * ((res['+dm'].ewm(com=14 - 1).mean()) / res['atr'])
        res['-di'] = 100 * ((res['-dm'].ewm(com=14 - 1).mean()) / res['atr'])
        res['adx'] = 100 * ((res['+di'] - res['-di']).apply(abs) / (res['+di'] + res['-di'])).ewm(com=14 - 1).mean()

        res.drop(['tr', 'atr', '+dm', '-dm', 'upmoves', 'lowmoves'], axis=1, inplace=True)
        return res, ['b-', 'g--', 'r--', 'r']

    @staticmethod
    def plot_adx(p_data):
        fig, ax = plt.subplots(2, 1)

        p_data[p_data.columns[0]].plot(ax=ax[0])
        ax[0].set_ylabel('Close value')
        ax[0].yaxis.tick_right()

        p_data['adx'].plot(ax=ax[1])
        high_mark = 50 * np.ones(p_data.index.size)
        low_mark = 20 * np.ones(p_data.index.size)
        ax[1].plot(p_data.index, high_mark)
        ax[1].plot(p_data.index, low_mark)
        ax[1].fill_between(p_data.index, high_mark, low_mark, alpha=_alpha, facecolor='red')
        ax[1].set_ylabel('Stochastic indicator')
        ax[1].legend()


class StockAnalyzer:

    __version__ = 0.1

    def __init__(self, session_name):
        self.session_name = session_name
        self.input_string = "("+self.session_name+")>> "
        self.default_ana_dur = timedelta(6*365/12)
        self.end = datetime.now()
        self.auto_save = True
        self.stocks = dict()
        self.save_name = session_name

        self.parser = StrArgParser("Stock analyzer")
        self.add_commands()

        while True:
            start_date = input(self.input_string+"Enter start date of analysis (YYYY-MM-DD) {enter 'exit' to get "
                                                 "default analysis duration}: ")
            if start_date == 'exit':
                start_date = self.end - self.default_ana_dur
                print('Default analysis duration has been set')
                break
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                break
            except ValueError:
                print("Wrong date")

        self.start = start_date
        for s in self.stocks.values():
            s.fill_hist_data(self.start, self.end)

        self.private_params = ['private_params', 'input_string', 'stocks', 'parser']

    def update_stock_details(self, force=False):
        if self.end.date() != datetime.now().date() or force:
            self.end = datetime.now()
            for s in self.stocks.values():
                print("Updating: "+s.get_name())
                s.fill_hist_data(self.start, self.end)

    def add_commands(self):
        self.parser.add_command('add', "Command to add stocks for analysis")
        self.parser.get_command('add').add_optional_arguments('-sn', "--stock_name", "The stock name")
        self.parser.get_command('add').add_optional_arguments('-tr', "--tracker", "The tracker of the stock")

        self.parser.add_command('ls_stocks', "Lists all the stock added")

        self.parser.add_command('help', "Shows the details of all the commands")

        self.parser.add_command('save_session', "Saves the current session to a file")
        self.parser.get_command('save_session').add_compulsory_arguments('-fn', '--file_name',
                                                                         "The name/address of the "
                                                                         "file where the session "
                                                                         "is to be saved")
        self.parser.get_command('save_session').add_optional_arguments('-as', '--auto_save',
                                                                       "Sets the auto save true/false. By default "
                                                                       "false. Only accept 'true' or 'false' "
                                                                       "if any other value provided then command is "
                                                                       "neglected", param_type=bool)

        self.parser.add_command('ls_params', 'List the session parameters')

        self.parser.add_command('ch_params', "Update session parameters", inf_positional=True)

        self.parser.add_command('update_stocks', 'Updates the stock data', inf_positional=True)

        self.parser.add_command('exit', 'Exits the session')

        self.parser.add_command("version", "Displays the version of the program")

        self.parser.add_command('export_data', "Exports the raw data to the file. Use it to transition from one "
                                               "version to another")

    def cmd_add_stock(self, res):
        if len(res) >= 2:
            stock_name = res['-sn']
            stock_tracker = res['-tr']
        else:
            stock_name = input(self.input_string+"Enter the stock name: ")
            stock_tracker = input(self.input_string+"Enter the stock tracker: ")

        try:
            print("Fetching stock details")
            stock = Stock(stock_name, stock_tracker)
        except IndexError:
            print("Tracker name is wrong. Exiting")
            return
        except:
            print("Connection error. Exiting")
            return ""

        stock.fill_hist_data(self.start, self.end)
        self.stocks[stock_name] = stock
        print(stock)
        print("Stock added")

    def cmd_ls_stock(self):
        if len(self.stocks) == 0:
            print("Empty")
        else:
            for s in self.stocks.values():
                print(s)

    def cmd_show_help(self):
        self.parser.show_help()

    def cmd_save_session(self, res):
        fn = res['-fn']
        with open(fn, 'wb') as f:
            if len(res) > 1:
                self.auto_save = res['-as']
                self.save_name = res['-fn']
            pk.dump(self, f)
            print("Session saved to the file "+fn)

    def cmd_ls_params(self):
        params = self.__dict__
        val = [type(i) for i in list(params.values())]
        i = 0
        for k, v in params.items():
            if k not in self.private_params:
                print(k + "\t" + str(val[i]).replace('<class ', "").replace(">", "") + "\t\t" + str(v))
            i += 1

    def cmd_ch_params(self, res):
        param_dict = self.__dict__
        state = 'param'
        param = ""
        is_skip = False
        for v in res.values():
            if is_skip:
                is_skip = False
                continue
            if state == 'param':
                if v in self.private_params:
                    print("'"+v + "' is wrong parameter")
                    is_skip = True
                    continue
                param = v
                state = 'value'
            elif state == 'value':
                val = v
                val_error_string = ""
                try:
                    if type(param_dict[param]) is datetime:
                        val_error_string = ". Value should be in format YYYY-MM-DD"
                        val = datetime.strptime(val, '%Y-%m-%d')
                    elif type(param_dict[param]) is timedelta:
                        val_error_string = ". Value should be an integer"
                        val = timedelta(int(val))
                    elif type(param_dict[param]) is bool:
                        if val == 'true':
                            val = True
                        elif val == 'false':
                            val = False
                        else:
                            val_error_string = ". Value should be either 'true' or 'false'"
                            raise ValueError
                    else:
                        val_error_string = ". Value should be {0}".format(
                            str(type(param_dict[param])).replace('<class ', "").replace(">", ""))
                        val = type(param_dict[param])(val)
                except ValueError:
                    print("Value for the parameter '" + param + "' is wrong"+val_error_string)
                    continue
                except KeyError:
                    print("Parameter '"+param+"' not found")
                    continue
                finally:
                    state = 'param'
                self.__setattr__(param, val)
            else:
                print("Unexpected error has occurred")
                break

    def cmd_update_stocks(self, res):
        if len(res) == 0:
            self.update_stock_details(force=True)
        else:
            for v in res.values():
                print("Updating: "+self.stocks[v].get_name())
                self.stocks[v].fill_hist_data(self.start, self.end)

    def cmd_version(self):
        print("Session: "+self.session_name+" Version: "+str(StockAnalyzer.__version__))

    def cmd_exit(self):
        if self.auto_save:
            self.cmd_save_session({'-fn': self.save_name})
            print('Auto-saving...')

    def start_command_line(self):
        while True:
            s = (input(self.input_string).strip(' '))
            (cmd, res) = self.parser.decode_command(s)
            if res is None:
                continue
            if cmd == 'add':
                self.cmd_add_stock(res)
            elif cmd == 'ls_stocks':
                self.cmd_ls_stock()
            elif cmd == 'help':
                self.cmd_show_help()
            elif cmd == 'save_session':
                self.cmd_save_session(res)
            elif cmd == 'cmd_list':
                self.parser.show_cmd_list(res)
            elif cmd == 'ls_params':
                self.cmd_ls_params()
            elif cmd == 'ch_params':
                self.cmd_ch_params(res)
            elif cmd == 'update_stocks':
                self.cmd_update_stocks(res)
            elif cmd == 'version':
                self.cmd_version()
            elif cmd == 'export_data':
                pass
            elif cmd == 'exit':
                self.cmd_exit()
                break

    @staticmethod
    def __update_parameter__(stock, parameter):
        if stock.tracker in parameter:
            return parameter
        else:
            return stock.tracker + "_" + parameter

    # def get_clts(self, parameter):
    #     parameter2 = StockAnalyzer.__update_parameter__(self.stocks[0], parameter)
    #     result = pd.DataFrame(self.stocks[0].get_clt(parameter2))
    #     if len(self.stocks) > 1:
    #         for s in self.stocks[1:]:
    #             parameter2 = StockAnalyzer.__update_parameter__(s, parameter)
    #             result = result.join(s.get_clt(parameter2))
    #     return result

    # def get_price(self, parameter):
    #     parameter2 = StockAnalyzer.__update_parameter__(self.stocks[0], parameter)
    #     result = pd.DataFrame(self.stocks[0].get_data(parameter2))
    #     if len(self.stocks) > 1:
    #         for s in self.stocks[1:]:
    #             parameter2 = StockAnalyzer.__update_parameter__(s, parameter)
    #             result = result.join(s.get_data(parameter2))
    #     return result

    def resample_hist_data(self, sample_rate):
        for s in self.stocks.values():
            s.hist_data = s.hist_data.resample(sample_rate).last()
