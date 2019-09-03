from stock import Stock, StockAnalyzer
from strargparser import StrArgParser
import pickle as pk
from matplotlib import pyplot as plt
import datetime
import threading as threading

# infosys_stock = Stock('Insfosys', 'infy')
# reliance = Stock('Reliance', 'RELIANCE')
# icici_bank = Stock('ICICI_Bank', 'ICICIBANK')
# bajaj = Stock('Bajaj_auto', 'BAJAJ-AUTO')
# yes_bank = Stock('Yes_Bank', 'YESBANK')
#
# start_date = '2018-01-24'
# end_date = datetime.datetime.now()
#
# ana = StockAnalyzer([icici_bank, reliance, infosys_stock, bajaj, yes_bank], start_date, end_date)
#
# k, style = yes_bank.get_bollinger_bonds_indicator(20, 2)
# Stock.plot_bollinger_bonds(k, 20)

# kk, style = infosys_stock.get_ichimoku_kinko_hyo_indicator()
# Stock.plot_ichimoku_kinko_hyo(kk, style)

# k2, style = yes_bank.get_moving_average_indicator(20, 100)
# Stock.plot_moving_average(k2, 20, 100)

# k, style = yes_bank.get_rsi_indicator()
# Stock.plot_rsi(k)

# k, style = reliance.get_macd_indicator()
# Stock.plot_macd(k)

# k, style = yes_bank.get_parabolic_sar_indicator()
# Stock.plot_sar(k, style)

# k, style = icici_bank.get_adx_indicator()
# Stock.plot_adx(k)
# k, style = icici_bank.get_stochastic_indicator()
# Stock.plot_stochastic(k)

# plt.show()
par = StrArgParser("Main command parser")


def add_commands():
    par.add_command("exit", "Ends the program")

    par.add_command("start_session", "Starts a new session")
    par.get_command('start_session').add_compulsory_arguments('-sn', '--session_name', 'Session name')

    par.add_command('load_session', "Loads a previously saved session")
    par.get_command('load_session').add_compulsory_arguments('-fn', '--file_name',
                                                             'The name/address of the file from '
                                                             'which the session is to be loaded')

    par.add_command('help', "Lists all the commands of session management")


def handle_init():
    while True:
        s = input(">> ").strip(' ')
        (cmd, res) = par.decode_command(s)
        if res is None:
            continue
        elif cmd == 'exit':
            print('Exiting')
            exit(0)
        elif cmd == 'start_session':
            ana = StockAnalyzer(res['-sn'])
            ana.start_command_line()

        elif cmd == 'load_session':
            try:
                with open(res['-fn'], 'rb') as f:
                    ana = pk.load(f)
                    print('Session loaded. Updating stock data...')
                    ana.update_stock_details()
                    print('Stock data updated. Starting session')
                    ana.start_command_line()
            except FileNotFoundError:
                print('File not found')
        elif cmd == 'cmd_list':
            par.show_cmd_list(res)
        elif cmd == 'help':
            par.show_help()


if __name__ == '__main__':
    add_commands()
    handle_init()





