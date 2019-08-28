from stock import Stock, StockAnalyzer
from matplotlib import pyplot as plt
import datetime

infosys_stock = Stock('Insfosys', 'infy')
reliance = Stock('Reliance', 'RELIANCE')
icici_bank = Stock('ICICI_Bank', 'ICICIBANK')
bajaj = Stock('Bajaj_auto', 'BAJAJ-AUTO')

start_date = '2018-01-24'
end_date = datetime.datetime.now()

ana = StockAnalyzer([icici_bank, reliance, infosys_stock, bajaj], start_date, end_date)

# k, style = icici_bank.get_bollinger_bonds_indicator(20, 2)
# Stock.plot_bollinger_bonds(k, style, 20)

# kk, style = infosys_stock.get_ichimoku_kinko_hyo_indicator()
# Stock.plot_ichimoku_kinko_hyo(kk, style)

# k2, style = icici_bank.get_moving_average_indicator(20, 100)
# Stock.plot_moving_average(k2, style, 20, 100)

# k, style = reliance.get_rsi_indicator()
# Stock.plot_rsi(k)

# k, style = reliance.get_macd_indicator()
# Stock.plot_macd(k)

# k, style = reliance.get_parabolic_sar_indicator()
# Stock.plot_sar(k, style)

icici_bank.get_adx_indicator()
k, style = icici_bank.get_stochastic_indicator()
Stock.plot_stochastic(k)
plt.show()
