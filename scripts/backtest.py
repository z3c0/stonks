import sys
import json
import urllib
import numpy as np
import pandas as pd
import requests
import configparser
from scipy.optimize import brute
from pylab import mpl, plt
from queue import Queue
from threading import Thread
from multiprocessing import cpu_count

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

NUMBER_OF_THREADS = cpu_count()

PRINT_DIVIDER = '=' * 80 + '\n'

FIXED_TRAN_COST = 0.00
PROP_TRAN_COST = 0.00


class Backtest:
    def __init__(
        self,
        symbol,
        start,
        end,
        amount,
        fixed_tx_cost=0.0,
        prop_tx_cost=0.0,
        verbose=True,
    ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.initial_amount = amount
        self.fixed_tx_cost = fixed_tx_cost
        self.prop_tx_cost = prop_tx_cost
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.optimizing = False
        self.statitics = dict()

    def set_data(self, df, target_col):
        raw = pd.DataFrame(df[target_col])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={target_col: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def plot_data(self, cols=None):
        if cols is None:
            cols = ['price']
        self.data[cols].plot(figsize=(10, 6), title=self.symbol)

    def get_date_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price = self.data['price'].iloc[bar]
        return date, price

    def print_balance(self, bar):
        date, _ = self.get_date_price(bar)
        print(f'{date} | current balance {self.amount:.2f}')

    def print_net_wealth(self, bar):
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date} | current net wealth {net_wealth:.2f}')

    def place_buy_order(self, bar, units=None, amount=None):
        date, price = self.get_date_price(bar)

        if units is None:
            units = int(amount / price)

        self.amount -= \
            (units * price) * (1 + self.prop_tx_cost) + self.fixed_tx_cost

        self.units += units
        self.trades += 1

        if self.verbose and not self.optimizing:
            print(f'{date} | selling {units} units at price {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)

        self.amount += \
            (units * price) * (1 - self.prop_tx_cost) - self.fixed_tx_cost

        self.units -= units
        if self.verbose and not self.optimizing:
            print(f'{date} | selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        date, price = self.get_date_price(bar)
        self.amount += self.units * price

        self.units = 0
        self.trades += 1
        if self.verbose and not self.optimizing:
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print(PRINT_DIVIDER)

        performance = (self.amount - self.initial_amount) / self.initial_amount

        if not self.optimizing:
            print(f'Final balance\t[$] {self.amount:.2f}')
            print(f'Net Performance\t[%] {performance:.2%}')
            print(f'Trades Executed\t[#] {self.trades}')
            print(PRINT_DIVIDER)

        return performance

    @staticmethod
    def optimize(first_range, second_range, func):
        if first_range and second_range:
            optimization_range = (first_range, second_range)
        elif first_range:
            optimization_range = (first_range, )
        optimization = brute(func, optimization_range, finish=None)

        return optimization


class BacktestLong(Backtest):
    def run_sma_strategy(self, params=None):
        if params is None:
            self.optimizing = True
            params = Backtest.optimize((1, 42), (43, 252),
                                       self.run_sma_strategy)
            self.optimizing = False

        sma1 = int(params[0])
        sma2 = int(params[1])

        if not self.optimizing:
            print((f'{self.symbol}: long SMA strategy | '
                   f'SMA1={sma1}, SMA2={sma2}'))
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['SMA1'] = self.data['price'].rolling(sma1).mean()
        self.data['SMA2'] = self.data['price'].rolling(sma2).mean()

        for bar in range(sma2, len(self.data)):
            if self.position == 0:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if self.data['SMA2'].iloc[bar] > self.data['SMA1'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0

        performance = self.close_out(bar)
        self.statitics['sma'] = {
            'sma1': sma1,
            'sma2': sma2,
            'performance': performance
        }

        return -performance

    def run_momentum_strategy(self, momentum=None):
        if momentum is None:
            self.optimizing = True
            momentum = Backtest.optimize((1, 63), None,
                                         self.run_momentum_strategy)
            self.optimizing = False

        momentum = int(momentum)

        if not self.optimizing:
            print((f'{self.symbol}: long momentum strategy | '
                   f'{momentum} days'))
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()

        for bar in range(momentum, len(self.data)):
            if self.position == 0:
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if 0 > self.data['momentum'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0

        performance = self.close_out(bar)

        self.statitics['momentum'] = {
            'momentum': momentum,
            'performance': performance
        }

        return -performance

    def run_mean_reversion_strategy(self, params=None):
        if params is None:
            self.optimizing = True
            params = Backtest.optimize((21, 252), (1, 15),
                                       self.run_mean_reversion_strategy)
            self.optimizing = False

        sma = int(params[0])
        threshold = int(params[1])

        if not self.optimizing:
            print((f'{self.symbol}: long mean reversion strategy | '
                   f'SMA1={sma}, thr={threshold}'))
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['SMA'] = self.data['price'].rolling(sma).mean()

        for bar in range(sma, len(self.data)):
            if self.position == 0:
                if (self.data['SMA'].iloc[bar] - threshold >
                        self.data['price'].iloc[bar]):
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0

        performance = self.close_out(bar)

        self.statitics['mean_reversion'] = {
            'sma': sma,
            'threshold': threshold,
            'performance': performance
        }

        return -performance


class BacktestLongShort(Backtest):
    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.place_buy_order(bar, units=-self.units)

        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount

            self.place_buy_order(bar, amount=amount)

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)

        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount

            self.place_sell_order(bar, amount=amount)

    def run_sma_strategy(self, params=None):
        if params is None:
            self.optimizing = True
            params = Backtest.optimize((1, 42), (43, 252),
                                       self.run_sma_strategy)
            self.optimizing = False

        sma1 = int(params[0])
        sma2 = int(params[1])

        if not self.optimizing:
            print((f'{self.symbol}: long/short SMA strategy | '
                   f'SMA1={sma1}, SMA2={sma2}'))
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['SMA1'] = self.data['price'].rolling(sma1).mean()
        self.data['SMA2'] = self.data['price'].rolling(sma2).mean()

        for bar in range(sma2, len(self.data)):
            if 0 >= self.position:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.go_long(bar, amount='all')
                    self.position = 1

            if self.position >= 0:
                if self.data['SMA2'].iloc[bar] > self.data['SMA1'].iloc[bar]:
                    self.go_short(bar, amount='all')
                    self.position = -1

        performance = self.close_out(bar)
        self.statitics['sma'] = {
            'sma1': sma1,
            'sma2': sma2,
            'performance': performance
        }

        return -performance

    def run_momentum_strategy(self, momentum=None):
        if momentum is None:
            self.optimizing = True
            momentum = Backtest.optimize((1, 63), None,
                                         self.run_momentum_strategy)
            self.optimizing = False

        momentum = int(momentum)

        if not self.optimizing:
            print((f'{self.symbol}: long/short momentum strategy | '
                   f'{momentum} days'))
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()

        for bar in range(momentum, len(self.data)):

            if 0 >= self.position:
                if self.data['momentum'].iloc[bar] > 0:
                    self.go_long(bar, amount='all')
                    self.position = 1

            if self.position >= 0:
                if 0 >= self.data['momentum'].iloc[bar]:
                    self.go_short(bar, amount='all')
                    self.position = -1

        performance = self.close_out(bar)

        self.statitics['momentum'] = {
            'momentum': momentum,
            'performance': performance
        }

        return -performance

    def run_mean_reversion_strategy(self, params=None):
        if params is None:
            self.optimizing = True
            params = Backtest.optimize((21, 252), (1, 15),
                                       self.run_mean_reversion_strategy)
            self.optimizing = False

        sma = int(params[0])
        threshold = int(params[1])

        if not self.optimizing:
            msg = 'long/short mean reversion strategy'
            print(f'{self.symbol}: {msg} | SMA1={sma}, thr={threshold}')
            tx_cost_txt = (f'fixed costs {self.fixed_tx_cost} | '
                           f'proportional costs {self.prop_tx_cost}')
            print(tx_cost_txt)
            print(PRINT_DIVIDER)

        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['SMA'] = self.data['price'].rolling(sma).mean()

        for bar in range(sma, len(self.data)):
            if self.position == 0:
                if (self.data['SMA'].iloc[bar] - threshold >
                        self.data['price'].iloc[bar]):
                    self.go_long(bar, amount=self.initial_amount)
                    self.position = 1
                elif (self.data['price'].iloc[bar] >
                      self.data['SMA'].iloc[bar] + threshold):
                    self.go_short(bar, amount=self.initial_amount)
                    self.position = -1

            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0

            elif self.position == -1:
                if self.data['SMA'].iloc[bar] >= self.data['price'].iloc[bar]:
                    self.place_buy_order(bar, units=-self.units)
                    self.position = 0

        performance = self.close_out(bar)

        self.statitics['mean_reversion'] = {
            'sma': sma,
            'threshold': threshold,
            'performance': performance
        }

        return -performance


def run_backtests(target_symbol):
    target_symbol = target_symbol.replace('.', '-')
    target_col = 'Close'

    amount = 10000

    try:
        datasource_df = pd.read_csv(f'stonks/{target_symbol}.csv',
                                    index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None, None

    if 253 > len(datasource_df):
        return None, None

    start, end = min(datasource_df.index), max(datasource_df.index)
    long_backtest = BacktestLong(target_symbol, start, end, amount,
                                 FIXED_TRAN_COST, PROP_TRAN_COST, False)
    longshort_backtest = BacktestLongShort(target_symbol, start, end, amount,
                                           FIXED_TRAN_COST, PROP_TRAN_COST,
                                           False)

    long_backtest.set_data(datasource_df, target_col)
    longshort_backtest.set_data(datasource_df, target_col)

    long_backtest.run_sma_strategy()
    long_backtest.run_momentum_strategy()
    long_backtest.run_mean_reversion_strategy()

    longshort_backtest.run_sma_strategy()
    longshort_backtest.run_momentum_strategy()
    longshort_backtest.run_mean_reversion_strategy()

    return long_backtest, longshort_backtest


def download_stonk_list():
    page_number = 1
    records = list()

    total_records = 1

    while total_records > len(records):

        reponse = requests.post(
            'https://www.nyse.com/api/quotes/filter',
            json={
                'instrumentType': 'EQUITY',
                'pageNumber': page_number,
                'sortColumn': 'NORMALIZED_TICKER',
                'sortOrder': 'ASC',
                'maxResultsPerPage': 1000,
                'filterToken': '',
            },
        )

        response_data = json.loads(reponse.text)
        total_records = response_data[0]['total']

        records += list(response_data)
        page_number += 1

    nyse_df = pd.DataFrame(records)
    nyse_df.to_csv('NYSE.csv', index=False)


def download_stonk_history(target_symbol):
    target_symbol = target_symbol.replace('.', '-')

    config = configparser.ConfigParser()
    config.read('algotrade.cfg')

    API_KEY = config['eodhistoricaldata']['api_key']

    datasource = f'stonks/{target_symbol}.csv'

    endpoint = (f'https://eodhistoricaldata.com/api/eod/{target_symbol}.US'
                f'?api_token={API_KEY}')

    print(f'Downloading {target_symbol} history')

    try:

        ticker_eod = pd.read_csv(endpoint, index_col=0, parse_dates=True)
        ticker_eod.drop(ticker_eod.tail(1).index, inplace=True)
        ticker_eod['Symbol'] = target_symbol
        ticker_eod.to_csv(datasource)
    except urllib.error.HTTPError:
        print(f'404 encountered for {target_symbol}')


def download_nyse_history():
    nyse_df = pd.read_csv('NYSE.csv', index_col=5)

    def _download_stonk_concurrently():
        keep_threading = True
        while keep_threading:
            ticker = q.get()

            if ticker:
                download_stonk_history(ticker)
            else:
                keep_threading = False

            q.task_done()

    q = Queue(NUMBER_OF_THREADS * 2)
    for _ in range(NUMBER_OF_THREADS):
        thr = Thread(target=_download_stonk_concurrently, daemon=True)
        thr.start()

    try:
        for ticker in nyse_df.index:
            if nyse_df['instrumentType'].loc[ticker] != 'COMMON_STOCK':
                continue

            q.put(ticker)

        q.join()
    except KeyboardInterrupt:
        sys.exit(0)


def backtest_possible_strategies():

    print('Optimizing strategies')
    print(PRINT_DIVIDER)

    backtest_results = list()

    def _save_progress():
        data = [processed_backtests, pd.DataFrame(backtest_results)]
        backtests_df = pd.concat(data, axis=0)
        backtests_df.reset_index(drop=True, inplace=True)
        backtests_df.to_json('backtest_results.json')

    def _run_backtests_concurrently():
        keep_threading = True
        while keep_threading:
            ticker = q.get()

            if ticker:
                long, long_short = run_backtests(ticker)
                if long:
                    results = {'symbol': ticker,
                               'long': long.statitics,
                               'long_short': long_short.statitics}

                    backtest_results.append(results)
            else:
                keep_threading = False

            # save data periodically
            results_count = len(backtest_results)
            if results_count % 2 == 0 and results_count > 0:
                print(f'Checkpoint reached ({results_count})')
                print(PRINT_DIVIDER)
                _save_progress()

            q.task_done()

        print('Exiting thread')
        print(PRINT_DIVIDER)

    q = Queue(NUMBER_OF_THREADS * 2)

    print(f'Creating {NUMBER_OF_THREADS} threads')
    print(PRINT_DIVIDER)
    for _ in range(NUMBER_OF_THREADS):
        thr = Thread(target=_run_backtests_concurrently, daemon=True)
        thr.start()

    nyse_df = pd.read_csv('NYSE.csv', index_col=5)

    try:
        processed_backtests = pd.read_json('backtest_results.json')
        processed_tickers = set(processed_backtests['symbol'].unique())
    except OSError:
        with open('backtest_results.json', 'w') as f:
            f.write('{}')

        processed_backtests = pd.DataFrame([])
        processed_tickers = set()

    try:
        for ticker in nyse_df.index:
            if nyse_df['instrumentType'].loc[ticker] != 'COMMON_STOCK':
                continue

            if ticker in processed_tickers:
                continue

            q.put(ticker)

            print(f'Queued {ticker} for processing')
            print(PRINT_DIVIDER)
        q.join()
    except KeyboardInterrupt:
        _save_progress()
        sys.exit(0)

    print('Saving data')
    print(PRINT_DIVIDER)
    _save_progress()

    print('Sending exit signals to threads')
    print(PRINT_DIVIDER)
    for _ in range(NUMBER_OF_THREADS):
        q.put(False)

    q.join()

    print('Processing complete')
    print(PRINT_DIVIDER)


def process_backtest_results():
    print('Processing strategies')
    print(PRINT_DIVIDER)

    strategies = list()
    for result in json.load('backtest_results.json'):
        optimal_strategies = {'ticker': result['symbol']}

        if result['long']['sma']['performance'] > 0:
            optimal_strategies['l_sma1'] = result['long']['sma']['sma1']
            optimal_strategies['l_sma2'] = result['long']['sma']['sma2']
            optimal_strategies['l_sma_perf'] = \
                result['long']['sma']['performance']

        if result['long']['momentum']['performance'] > 0:
            optimal_strategies['l_m'] = \
                result['long']['momentum']['momentum']
            optimal_strategies['l_m_perf'] = \
                result['long']['momentum']['performance']

        if result['long']['mean_reversion']['performance'] > 0:
            optimal_strategies['l_mr_sma'] = \
                result['long']['mean_reversion']['sma']
            optimal_strategies['l_mr_thr'] = \
                result['long']['mean_reversion']['threshold']
            optimal_strategies['l_mr_perf'] = \
                result['long']['mean_reversion']['performance']

        if result['long_short']['sma']['performance'] > 0:
            optimal_strategies['ls_sma1'] = \
                result['long_short']['sma']['sma1']
            optimal_strategies['ls_sma2'] = \
                result['long_short']['sma']['sma2']

        if result['long_short']['momentum']['performance'] > 0:
            optimal_strategies['ls_m'] = \
                result['long_short']['momentum']['momentum']
            optimal_strategies['ls_m_perf'] = \
                result['long_short']['momentum']['performance']

        if result['long_short']['mean_reversion']['performance'] > 0:
            optimal_strategies['ls_mr_sma'] = \
                result['long_short']['mean_reversion']['sma']
            optimal_strategies['ls_mr_threshold'] = \
                result['long_short']['mean_reversion']['threshold']
            optimal_strategies['ls_mr_perf'] = \
                result['long_short']['mean_reversion']['performance']

        strategies.append(optimal_strategies)

    strategy_df = pd.DataFrame(strategies)
    strategy_df.to_csv('optimized_strategies.csv', index=False)


if __name__ == '__main__':
    # download_nyse_history()
    backtest_possible_strategies()
    process_backtest_results()
