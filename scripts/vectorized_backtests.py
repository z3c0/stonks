import random
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.optimize import brute
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class VectorizedBacktester:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end

        self.results = None

    def set_data(self, df, target_col):
        raw = pd.DataFrame(df[target_col])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={target_col: self.symbol}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        self.data = raw

    def run_strategy(self):
        pass

    def plot_results(self):
        pass


class SimpleMovingAverage(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Simple Moving Average'

    def run_strategy(self):
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)

        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)

        self.results = data

        gross_performance = data['cstrategy'].iloc[-1]

        outperformance = gross_performance - data['creturns'].iloc[-1]

        return round(gross_performance, 2), round(outperformance, 2)

    def plot_results(self):
        if self.results is None:
            self.run_strategy()

        plot_kwargs = {
            'title': (f'{self.symbol} | ' + f'{self.title} | ' +
                      f'SMA1={self.first_sma}, ' + f'SMA2={self.second_sma}'),
            'figsize': (10, 6),
        }

        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)

    def set_parameters(self, first_sma=None, second_sma=None):
        if first_sma is not None:
            self.first_sma = first_sma

        if second_sma is not None:
            self.second_sma = second_sma

        self.data['SMA1'] = self.data[self.symbol].rolling(
            self.first_sma).mean()
        self.data['SMA2'] = self.data[self.symbol].rolling(
            self.second_sma).mean()

    def update_and_run(self, sma):
        self.set_parameters(int(sma[0]), int(sma[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, first_sma_range, second_sma_range):
        optimization_range = (first_sma_range, second_sma_range)
        optimization = brute(self.update_and_run,
                             optimization_range,
                             finish=None)

        return optimization, -self.update_and_run(optimization)


class Momentum(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Momentum'

    def set_parameters(self, amount, tran_cost, momentum=None):
        self.amount = amount
        self.tran_cost = tran_cost

        if momentum is not None:
            self.momentum = momentum

    def update_and_run(self, momentum):
        try:
            momentum = int(momentum[0])
        except IndexError:
            momentum = int(momentum)

        self.set_parameters(self.amount, self.tran_cost, momentum)
        return -self.run_strategy()[0]

    def optimize_parameters(self, amount, tran_cost, momentum_range):
        self.set_parameters(amount, tran_cost, 1)
        optimization = brute(self.update_and_run, (momentum_range, ),
                             finish=None)

        return optimization, -self.update_and_run(optimization)

    def run_strategy(self):
        data = self.data.copy().dropna()

        data['position'] = np.sign(data['return'].rolling(
            self.momentum).mean())
        data['strategy'] = data['position'].shift(1) * data['return']

        data.dropna(inplace=True)
        trades = data['position'].diff().fillna(0) != 0

        # pay for transactions
        data['strategy'][trades] -= self.tran_cost

        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)

        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(
            np.exp)

        self.results = data

        absolute_performance = self.results['cstrategy'].iloc[-1]
        outperformance = absolute_performance - self.results['creturns'].iloc[
            -1]

        return round(absolute_performance, 2), round(outperformance, 2)

    def plot_results(self):
        if self.results is None:
            self.run_strategy(1)

        plot_kwargs = {
            'title': (f'{self.symbol} | '
                      f'{self.title} | '
                      f'TC={self.tran_cost:.4f}, '
                      f'P={self.momentum}'),
            'figsize': (10, 6),
        }
        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)


class MeanReversion(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Mean Reversion'

    def set_parameters(self, amount, tran_cost, sma=None, threshold=None):
        self.amount = amount
        self.tran_cost = tran_cost

        if sma is not None:
            self.sma = sma

        if threshold is not None:
            self.threshold = threshold

    def update_and_run(self, sma_threholds):
        self.set_parameters(self.amount, self.tran_cost, int(sma_threholds[0]),
                            int(sma_threholds[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, amount, tran_cost, sma_range,
                            threshold_range):
        optimization_range = (sma_range, threshold_range)
        self.set_parameters(amount, tran_cost, 1, 0.1)
        optimization = brute(self.update_and_run,
                             optimization_range,
                             finish=None)

        return optimization, -self.update_and_run(optimization)

    def run_strategy(self):
        data = self.data.copy().dropna()

        data['sma'] = data[self.symbol].rolling(self.sma).mean()
        data['distance'] = data[self.symbol] - data['sma']

        data.dropna(inplace=True)

        # sell
        data['position'] = np.where(data['distance'] > self.threshold, -1,
                                    np.nan)

        # buy
        data['position'] = np.where(data['distance'] < -self.threshold, 1,
                                    data['position'])

        # cross price and SMA (zero distance)
        data['position'] = np.where(
            data['distance'] * data['distance'].shift(1) < 0, 0,
            data['position'])

        data['position'] = data['position'].ffill().fillna(0)
        data['strategy'] = data['position'].shift(1) * data['return']

        trades = data['position'].diff().fillna(0) != 0

        # pay for transactions
        data['strategy'][trades] -= self.tran_cost

        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)

        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(
            np.exp)

        self.results = data

        absolute_performance = self.results['cstrategy'].iloc[-1]
        outperformance = absolute_performance - self.results['creturns'].iloc[
            -1]

        return round(absolute_performance, 2), round(outperformance, 2)

    def plot_results(self):
        if self.results is None:
            self.run_strategy(1)

        plot_kwargs = {
            'title': (f'{self.symbol} | '
                      f'{self.title} | '
                      f'TC={self.tran_cost:.4f}, '
                      f'SMA={self.sma}, '
                      f'THR={self.threshold}'),
            'figsize': (10, 6),
        }
        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)


class LinearRegression(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Regression'

    def set_data(self, df, target_col, train_col):
        if target_col == train_col:
            raw = pd.DataFrame(df[[target_col]])
            raw.rename(columns={target_col: self.symbol}, inplace=True)
            raw['TRAIN'] = raw[self.symbol]
        else:
            raw = pd.DataFrame(df[[target_col, train_col]])
            raw.rename(columns={
                target_col: self.symbol,
                train_col: 'TRAIN'
            },
                       inplace=True)

        raw = raw.loc[self.start:self.end]

        raw['return'] = np.log(raw[[self.symbol]] /
                               raw[[self.symbol]].shift(1))
        raw['returns_train'] = np.log(raw[['TRAIN']] / raw[['TRAIN']].shift(1))
        self.data = raw

    def set_parameters(self, amount, tran_cost):
        self.amount = amount
        self.tran_cost = tran_cost

    def select_data(self, start, end):
        selection = pd.DataFrame(self.data[(self.data.index >= start)
                                           & (self.data.index <= end)])
        return selection

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(
            np.dot(self.lagged_data[self.cols], self.regression))
        self.results['prediction'] = prediction
        self.results[
            'strategy'] = self.results['prediction'] * self.results['return']

        trades = self.results['prediction'].diff().fillna(0) != 0
        self.results['strategy'][trades] -= self.tran_cost

        self.results['creturns'] = self.amount * self.results['return'].cumsum(
        ).apply(np.exp)
        self.results['cstrategy'] = self.amount * self.results[
            'strategy'].cumsum().apply(np.exp)

        absolute_performance = self.results['cstrategy'].iloc[-1]
        outperformance = absolute_performance - self.results['creturns'].iloc[
            -1]

        return round(absolute_performance, 2), round(outperformance, 2)

    def prepare_lags(self, start, end):
        selection = self.select_data(start, end)
        self.cols = []

        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            selection[col] = selection['returns_train'].shift(lag)
            self.cols.append(col)

        selection.dropna(inplace=True)
        self.lagged_data = selection

    def fit_model(self, start, end):
        self.prepare_lags(start, end)
        regression = np.linalg.lstsq(self.lagged_data[self.cols],
                                     np.sign(self.lagged_data['return']),
                                     rcond=None)[0]

        self.regression = regression

    def plot_results(self):
        plot_kwargs = {
            'title': (f'{self.symbol} | '
                      f'{self.title} | '
                      f'TC={self.tran_cost:.4f}'),
            'figsize': (10, 6),
        }
        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)


class LogisticRegression(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Classification (LR)'

    def set_data(self, df, target_col, train_col):
        if target_col == train_col:
            raw = pd.DataFrame(df[[target_col]])
            raw.rename(columns={target_col: self.symbol}, inplace=True)
            raw['TRAIN'] = raw[self.symbol]
        else:
            raw = pd.DataFrame(df[[target_col, train_col]])
            raw.rename(columns={
                target_col: self.symbol,
                train_col: 'TRAIN'
            },
                       inplace=True)

        raw = raw.loc[self.start:self.end]

        raw['return'] = np.log(raw[[self.symbol]] /
                               raw[[self.symbol]].shift(1))
        raw['returns_train'] = np.log(raw[['TRAIN']] / raw[['TRAIN']].shift(1))
        self.data = raw

    def set_parameters(self, amount, tran_cost, model):
        self.amount = amount
        self.tran_cost = tran_cost
        if model == 'linear':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C=1e6,
                                                         solver='lbfgs',
                                                         multi_class='ovr',
                                                         max_iter=1000)
        else:
            raise ValueError('Unknown Model')

    def select_data(self, start, end):
        selection = pd.DataFrame(self.data[(self.data.index >= start)
                                           & (self.data.index <= end)])
        return selection

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        self.lags = lags
        self.fit_model(start_in, end_in)

        self.prepare_features(start_out, end_out)
        prediction = self.model.predict(self.data_subset[self.feature_columns])

        self.data_subset['prediction'] = prediction

        self.data_subset['strategy'] = (self.data_subset['prediction'] *
                                        self.data_subset['return'])

        trades = self.data_subset['prediction'].diff().fillna(0) != 0

        self.data_subset['strategy'] = np.where(
            trades,
            self.data_subset['strategy'] - self.tran_cost,
            self.data_subset['strategy'],
        )

        self.data_subset['creturns'] = self.amount * self.data_subset[
            'return'].cumsum().apply(np.exp)

        self.data_subset['cstrategy'] = self.amount * self.data_subset[
            'strategy'].cumsum().apply(np.exp)

        self.results = self.data_subset

        absolute_performance = self.results['cstrategy'].iloc[-1]
        outperformance = absolute_performance - self.results['creturns'].iloc[
            -1]

        return round(absolute_performance, 2), round(outperformance, 2)

    def fit_model(self, start, end):
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns],
                       np.sign(self.data_subset['return']))

    def prepare_features(self, start, end):
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []

        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            self.data_subset[col] = self.data_subset['returns_train'].shift(
                lag)
            self.feature_columns.append(col)

        self.data_subset.dropna(inplace=True)

    def plot_results(self):
        plot_kwargs = {
            'title': (f'{self.symbol} | '
                      f'{self.title} | '
                      f'TC={self.tran_cost:.4f}'),
            'figsize': (10, 6),
        }
        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)


class DenseNeuralNetwork(VectorizedBacktester):
    def __init__(self, symbol, start, end):
        super().__init__(symbol, start, end)
        self.title = 'Dense Neural Network'

    def set_data(self, df, target_col, train_col):
        if target_col == train_col:
            raw = pd.DataFrame(df[[target_col]])
            raw.rename(columns={target_col: self.symbol}, inplace=True)
            raw['TRAIN'] = raw[self.symbol]
        else:
            raw = pd.DataFrame(df[[target_col, train_col]])
            raw.rename(columns={
                target_col: self.symbol,
                train_col: 'TRAIN'
            },
                       inplace=True)

        raw = raw.loc[self.start:self.end]

        raw['return'] = np.log(raw[[self.symbol]] /
                               raw[[self.symbol]].shift(1))
        raw['returns_train'] = np.log(raw[['TRAIN']] / raw[['TRAIN']].shift(1))
        raw['direction'] = np.where(raw['returns_train'] > 0, 1, 0)
        self.data = raw

    def set_parameters(self, amount, tran_cost):
        self.amount = amount
        self.tran_cost = tran_cost

    def select_data(self, start, end):
        selection = pd.DataFrame(self.data[(self.data.index >= start)
                                           & (self.data.index <= end)])
        return selection

    def prepare_features(self, start, end):
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []

        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            self.data_subset[col] = self.data_subset['returns_train'].shift(
                lag)
            self.feature_columns.append(col)

        self.data_subset.dropna(inplace=True)

    def fit_model(self, start, end):
        self.prepare_features(start, end)

        optimizer = Adam(learning_rate=0.0001)
        set_seeds()
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.lags, )))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(
            self.data_subset[self.feature_columns],
            self.data_subset['direction'],
            epochs=50,
            verbose=False,
            validation_split=0.2,
            shuffle=False,
        )

        self.model = model

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        self.lags = lags

        self.fit_model(start_in, end_in)

        self.prepare_features(start_out, end_out)
        data = self.data_subset[self.feature_columns]
        prediction = np.where(self.model.predict(data) > 0.5, 1, 0)
        self.data_subset['prediction'] = np.where(prediction > 0, 1, -1)
        self.data_subset['strategy'] = (self.data_subset['prediction'] *
                                        self.data_subset['return'])

        trades = self.data_subset['prediction'].diff().fillna(0) != 0

        self.data_subset['strategy'] = np.where(
            trades,
            self.data_subset['strategy'] - self.tran_cost,
            self.data_subset['strategy'],
        )

        self.data_subset['creturns'] = self.amount * self.data_subset[
            'return'].cumsum().apply(np.exp)

        self.data_subset['cstrategy'] = self.amount * self.data_subset[
            'strategy'].cumsum().apply(np.exp)

        self.results = self.data_subset

        absolute_performance = self.results['cstrategy'].iloc[-1]
        outperformance = \
            absolute_performance - self.results['creturns'].iloc[-1]

        return round(absolute_performance, 2), round(outperformance, 2)

    def plot_results(self):
        plot_kwargs = {
            'title': (f'{self.symbol} | '
                      f'{self.title} | '
                      f'TC={self.tran_cost:.4f}'),
            'figsize': (10, 6),
        }
        self.results[['creturns', 'cstrategy']].plot(**plot_kwargs)


def test_backtest_classes():
    config = configparser.ConfigParser()
    config.read('algotrade.cfg')

    API_KEY = config['eodhistoricaldata']['api_key']

    target_symbol = 'BP'
    datasource = f'{target_symbol}.csv'

    endpoint = (f'https://eodhistoricaldata.com/api/eod/{target_symbol}.US'
                f'?api_token={API_KEY}')

    ticker_eod = pd.read_csv(endpoint, index_col=0, parse_dates=True)
    ticker_eod.drop(ticker_eod.tail(1).index, inplace=True)
    ticker_eod['Symbol'] = target_symbol
    ticker_eod.to_csv(datasource)

    start, end = '2010-1-1', '2021-12-31'
    train_start, train_end = '1981-1-1', '2009-12-31'
    target_col = 'Close'
    train_col = 'High'

    amount = 10000
    tran_cost = 0.001

    datasource_df = pd.read_csv(datasource, index_col=0, parse_dates=True)

    dataset = pd.DataFrame(
        datasource_df[datasource_df['Symbol'] == target_symbol])
    dataset = dataset.dropna()

    sma = SimpleMovingAverage(target_symbol, start, end)
    momentum = Momentum(target_symbol, start, end)
    mean_reversion = MeanReversion(target_symbol, start, end)
    linear_regression = LinearRegression(target_symbol, train_start, end)
    logistic_regression = LogisticRegression(target_symbol, train_start, end)
    # deep_learning = DenseNeuralNetwork(target_symbol, train_start, end)

    for test in [sma, momentum, mean_reversion]:
        test.set_data(dataset, target_col)

    for ml_test in [linear_regression, logistic_regression]:
        ml_test.set_data(dataset, target_col, train_col)

    sma.optimize_parameters((10, 59), (60, 508))
    momentum.optimize_parameters(amount, tran_cost, (1, 42))
    mean_reversion.optimize_parameters(amount, tran_cost, (21, 252), (1, 10))
    linear_regression.set_parameters(amount, tran_cost)
    logistic_regression.set_parameters(amount, tran_cost, 'logistic')
    # deep_learning.set_parameters(amount, tran_cost)

    training_dates = (train_start, train_end, start, end)

    output_str = 'absolute performance: {}\toutperformance: {}'
    sma_performance = sma.run_strategy()
    momentum_performance = momentum.run_strategy()
    mean_reversion_performance = mean_reversion.run_strategy()
    regression_performance = linear_regression.run_strategy(*training_dates,
                                                            lags=5)
    classification_performance = logistic_regression.run_strategy(
        *training_dates, lags=5)
    # deep_learning_performance = deep_learning.run_strategy(*training_dates,
    #                                                        lags=5)

    print(output_str.format(*sma_performance))
    print(output_str.format(*momentum_performance))
    print(output_str.format(*mean_reversion_performance))
    print(output_str.format(*regression_performance))
    print(output_str.format(*classification_performance))
    # print(output_str.format(*deep_learning_performance))

    recommend_sma = sma_performance[1] > 0

    recommend_momentum = momentum_performance[
        0] > amount or momentum_performance[1] > 0

    recommend_mean_reversion = (mean_reversion_performance[0] > amount
                                or mean_reversion_performance[1] > 0)

    recommend_regression = (regression_performance[0] > amount
                            or regression_performance[1] > 0)

    recommend_classification = (classification_performance[0] > amount
                                or classification_performance[1] > 0)

    # recommend_deep_learning = (deep_learning_performance[0] > amount
    #                            or deep_learning_performance[1] > 0)

    if recommend_sma:
        sma.plot_results()

    if recommend_momentum:
        momentum.plot_results()

    if recommend_mean_reversion:
        mean_reversion.plot_results()

    if recommend_regression:
        linear_regression.plot_results()

    if recommend_classification:
        logistic_regression.plot_results()

    # if recommend_deep_learning:
    #     deep_learning.plot_results()

    plt.show()
