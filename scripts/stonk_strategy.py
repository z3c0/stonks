import sys
import glob
import json
import math
import urllib
import requests
import numpy as np
import pandas as pd
import configparser
import pprint as pp
from pylab import mpl, plt
from queue import Queue
from threading import Thread
from multiprocessing import cpu_count

from backtest import BacktestLong, BacktestLongShort

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

config = configparser.ConfigParser()
config.read('algotrade.cfg')

ACCOUNT_BALANCE = 10000

FIXED_TRAN_COST = 0.00
PROP_TRAN_COST = 0.00

PRINT_DIVIDER = '=' * 80 + '\n'

NUMBER_OF_THREADS = cpu_count()


def run_backtests(target_symbol):
    target_symbol = target_symbol.replace('.', '-')
    target_col = 'Close'

    try:
        datasource_df = pd.read_csv(f'stonks/{target_symbol}.csv',
                                    index_col=0,
                                    parse_dates=True)
    except FileNotFoundError:
        return None, None

    if 253 > len(datasource_df):
        return None, None

    start, end = min(datasource_df.index), max(datasource_df.index)
    long_backtest = BacktestLong(target_symbol, start, end, ACCOUNT_BALANCE,
                                 FIXED_TRAN_COST, PROP_TRAN_COST, False)
    longshort_backtest = BacktestLongShort(target_symbol, start, end,
                                           ACCOUNT_BALANCE, FIXED_TRAN_COST,
                                           PROP_TRAN_COST, False)

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
                    results = {
                        'symbol': ticker,
                        'long': long.statitics,
                        'long_short': long_short.statitics
                    }

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
    strategies = list()

    backtest_results = pd.read_json('backtest_results.json')

    for _, row in backtest_results.iterrows():
        try:
            long = dict(row[1])
            long_short = dict(row[2])
        except IndexError:
            continue

        symbol = str(row[0])
        optimal_strategies = {'ticker': symbol}

        if long['sma']['performance'] > 0:
            optimal_strategies['l_sma1'] = long['sma']['sma1']
            optimal_strategies['l_sma2'] = long['sma']['sma2']
            optimal_strategies['l_sma_perf'] = long['sma']['performance']

        if long['momentum']['performance'] > 0:
            optimal_strategies['l_m'] = \
                long['momentum']['momentum']
            optimal_strategies['l_m_perf'] = \
                long['momentum']['performance']

        if long['mean_reversion']['performance'] > 0:
            optimal_strategies['l_mr_sma'] = \
                long['mean_reversion']['sma']
            optimal_strategies['l_mr_thr'] = \
                long['mean_reversion']['threshold']
            optimal_strategies['l_mr_perf'] = \
                long['mean_reversion']['performance']

        if long_short['sma']['performance'] > 0:
            optimal_strategies['ls_sma1'] = \
                long_short['sma']['sma1']
            optimal_strategies['ls_sma2'] = \
                long_short['sma']['sma2']

        if long_short['momentum']['performance'] > 0:
            optimal_strategies['ls_m'] = \
                long_short['momentum']['momentum']
            optimal_strategies['ls_m_perf'] = \
                long_short['momentum']['performance']

        if long_short['mean_reversion']['performance'] > 0:
            optimal_strategies['ls_mr_sma'] = \
                long_short['mean_reversion']['sma']
            optimal_strategies['ls_mr_threshold'] = \
                long_short['mean_reversion']['threshold']
            optimal_strategies['ls_mr_perf'] = \
                long_short['mean_reversion']['performance']

        strategies.append(optimal_strategies)

    strategy_df = pd.DataFrame(strategies)
    strategy_df.to_csv('optimized_strategies.csv', index=False)


def get_alpaca_account():

    headers = {
        'APCA-API-KEY-ID': config['alpaca']['paper_api_key'],
        'APCA-API-SECRET-KEY': config['alpaca']['paper_secret']
    }
    alpaca_account = \
        requests.get('https://paper-api.alpaca.markets/v2/account', headers)

    alpaca_account_json = json.loads(alpaca_account.text)
    pp.pprint(alpaca_account_json)


def process_close_prices():
    csv_glob = glob.glob('stonks/*.csv')

    final_records = list()
    for csv_path in csv_glob:
        csv_df = pd.read_csv(csv_path)
        final_record = csv_df.tail(1).iloc[:1].to_dict()

        try:
            final_record = {k: list(v.values())[0]
                            for k, v in final_record.items()}
            final_records.append(final_record)
        except IndexError:
            continue

    close_df = pd.DataFrame(final_records)
    close_df.to_csv('close_prices.csv', index=False)


def apply_optimal_strategies():

    close_prices_df = pd.read_csv('close_prices.csv', index_col=7)
    optimal_strategies_df = \
        pd.read_csv('optimized_strategies.csv', index_col=0)

    processed_strategies = list()

    for symbol, close_prices in close_prices_df.iterrows():
        if symbol in optimal_strategies_df.index:
            strategies = optimal_strategies_df.loc[symbol].to_dict()

            symbol_history = pd.read_csv(f'stonks/{symbol}.csv', index_col=0)
            price = close_prices['Close']

            symbol_history['Return'] = \
                np.log(symbol_history['Close']
                       / symbol_history['Close'].shift(1))

            if not math.isnan(strategies['l_sma_perf']):
                sma_short = int(strategies['l_sma1'])
                sma_long = int(strategies['l_sma2'])

                sma_short_value = \
                    (symbol_history.tail(sma_short))['Close'].mean()
                sma_long_value = \
                    (symbol_history.tail(sma_long))['Close'].mean()

                if sma_short_value > sma_long_value:
                    sma_action = 'Buy'
                elif sma_long_value > sma_short_value:
                    sma_action = 'Sell'
                else:
                    sma_action = 'Hold'
            else:
                sma_action = None

            if not math.isnan(strategies['l_m_perf']):
                momentum = int(strategies['l_m'])
                momentum_value = \
                    (symbol_history.tail(momentum))['Return'].mean()

                if momentum_value > 0:
                    momentum_action = 'Buy'
                elif 0 > momentum_value:
                    momentum_action = 'Sell'
                else:
                    momentum_action = 'Hold'
            else:
                momentum_action = None

            if not math.isnan(strategies['l_mr_perf']):
                sma = int(strategies['l_mr_sma'])
                threshold = int(strategies['l_mr_thr'])

                sma_value = (symbol_history.tail(sma))['Close'].mean()

                if sma_value - threshold > price:
                    mean_reversion_action = 'Buy'
                elif price >= sma_value:
                    mean_reversion_action = 'Sell'
                else:
                    mean_reversion_action = 'Hold'
            else:
                mean_reversion_action = None

            processed_strategies.append((symbol, price, sma_action,
                                         momentum_action,
                                         mean_reversion_action))

    processed_strategies_df = pd.DataFrame(processed_strategies)
    processed_strategies_df.columns = ('symbol', 'price', 'sma_action',
                                       'momentum_action',
                                       'mean_reversion_action')

    processed_strategies_df.to_csv('strategy_report.csv', index=False)


def build_superstonk_file():
    csv_glob = glob.glob('stonks/*.csv')

    first_pass = True

    open('superstonk.csv', 'w').close()
    for csv_path in csv_glob:
        csv_df = pd.read_csv(csv_path, index_col=0)
        if first_pass:
            csv_df.to_csv('superstonk.csv')
            first_pass = False
        else:
            csv_df.to_csv('superstonk.csv', mode='a', header=False)


if __name__ == '__main__':
    # print('refreshing stonks...')
    # download_nyse_history()

    # print('processing current prices...')
    # process_close_prices()

    # print('building superstonk file...')
    # build_superstonk_file()

    print('processing strategies...')
    process_backtest_results()

    print('applying strategies...')
    apply_optimal_strategies()

    print('stonk strategy complete')
