from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import Latest, CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns, RSI, RollingLinearRegressionOfReturns
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.filters import Q500US, Q1500US
from quantopian.pipeline.data.quandl import fred_usdontd156n as libor

import quantopian.experimental.optimize as opt

import talib
import pandas as pd
import numpy as np
from time import time
from collections import OrderedDict

from scipy import stats
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, svm, cross_validation


# Global configuration of strategy

N_STOCKS_TO_TRADE = 500
ML_TRAINING_WINDOW = 125
PRED_N_FWD_DAYS = 1
TRADE_FREQ = date_rules.every_day()

# Pipeline factors
bs = morningstar.balance_sheet
cfs = morningstar.cash_flow_statement
is_ = morningstar.income_statement
or_ = morningstar.operation_ratios
er = morningstar.earnings_report
v = morningstar.valuation
vr = morningstar.valuation_ratios

class Sector(Sector):
    window_safe = True

def make_factors():
    def Asset_Growth_3M():
        return Returns(inputs=[bs.total_assets], window_length=63)

    def Asset_To_Equity_Ratio():
        return bs.total_assets.latest / bs.common_stock_equity.latest

    def Capex_To_Cashflows():
        return (cfs.capital_expenditure.latest * 4.) / \
            (cfs.free_cash_flow.latest * 4.)

    def EBITDA_Yield():
        return (is_.ebitda.latest * 4.) / \
            USEquityPricing.close.latest

    def EBIT_To_Assets():
        return (is_.ebit.latest * 4.) / \
            bs.total_assets.latest

    def Return_On_Total_Invest_Capital():
        return or_.roic.latest

    class Mean_Reversion_1M(CustomFactor):
        inputs = [Returns(window_length=21)]
        window_length = 252

        def compute(self, today, assets, out, monthly_rets):
            out[:] = (monthly_rets[-1] - np.nanmean(monthly_rets, axis=0)) / \
                np.nanstd(monthly_rets, axis=0)

    class MACD_Signal_10d(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 60

        def compute(self, today, assets, out, close):

            sig_lines = []

            for col in close.T:
                try:
                    _, signal_line, _ = talib.MACD(col, fastperiod=12,
                                                   slowperiod=26, signalperiod=10)
                    sig_lines.append(signal_line[-1])
                except:
                    sig_lines.append(np.nan)
            out[:] = sig_lines

    class Moneyflow_Volume_5d(CustomFactor):
        inputs = [USEquityPricing.close, USEquityPricing.volume]
        window_length = 5

        def compute(self, today, assets, out, close, volume):
            mfvs = []
            for col_c, col_v in zip(close.T, volume.T):
                denominator = np.dot(col_c, col_v)
                numerator = 0.
                for n, price in enumerate(col_c.tolist()):
                    if price > col_c[n - 1]:
                        numerator += price * col_v[n]
                    else:
                        numerator -= price * col_v[n]
                mfvs.append(numerator / denominator)
            out[:] = mfvs

    def Net_Income_Margin():
        return or_.net_margin.latest

    def Operating_Cashflows_To_Assets():
        return (cfs.operating_cash_flow.latest * 4.) / \
            bs.total_assets.latest

    def Price_Momentum_3M():
        return Returns(window_length=63)

    class Price_Oscillator(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            four_week_period = close[-20:]
            out[:] = (np.nanmean(four_week_period, axis=0) /
                      np.nanmean(close, axis=0)) - 1.

    def Returns_39W():
        return Returns(window_length=215)

    class Trendline(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):

            X = range(self.window_length)
            X_bar = np.nanmean(X)
            X_vector = X - X_bar
            X_matrix = np.tile(X_vector, (len(close.T), 1)).T

            Y_bar = np.nanmean(close, axis=0)
            Y_bars = np.tile(Y_bar, (self.window_length, 1))
            Y_matrix = close - Y_bars

            X_var = np.nanvar(X)

            out[:] = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / \
                (self.window_length)

    class Vol_3M(CustomFactor):
        inputs = [Returns(window_length=2)]
        window_length = 63

        def compute(self, today, assets, out, rets):
            out[:] = np.nanstd(rets, axis=0)

    def Working_Capital_To_Assets():
        return bs.working_capital.latest / bs.total_assets.latest

    class AdvancedMomenutm(CustomFactor):
        """ Momentum factor """
        inputs = [USEquityPricing.close,
                  Returns(window_length=126)]
        window_length = 252

        def compute(self, today, assets, out, prices, returns):
            out[:] = ((prices[-21] - prices[-252])/prices[-252] -
                      (prices[-1] - prices[-21])/prices[-21]) / np.nanstd(returns, axis=0)

    def SPY_Beta():
        return RollingLinearRegressionOfReturns(
            target=symbols('SPY'),
            returns_length=2,
            regression_length=252
        ).beta

    all_factors = {
        'Mean Reversion 1M': Mean_Reversion_1M,
        'Price Momentum 3M': Price_Momentum_3M,
        'Price Oscillator': Price_Oscillator,
        '39 Week Returns': Returns_39W,
        'Trendline': Trendline,
        'Vol 3M': Vol_3M,
        'AdvancedMomenutm': AdvancedMomenutm,
    }

    return all_factors

def binarize_percentiles(Y, upper_percentile=70, lower_percentile=30):
    upper = np.nanpercentile(Y, upper_percentile, axis=1)[:, np.newaxis]
    lower = np.nanpercentile(Y, lower_percentile, axis=1)[:, np.newaxis]

    upper_mask = (Y >= upper)
    lower_mask = (Y <= lower)

    mask = upper_mask | lower_mask
    mask = mask.flatten()

    Y_binary = np.zeros(Y.shape[0] * Y.shape[1])
    Y_binary[upper_mask.flatten()] = 1
    Y_binary[lower_mask.flatten()] = -1

    return Y_binary, mask

def shift_mask_data(X, Y, upper_percentile=70, lower_percentile=30, n_fwd_days=1):
    shifted_X = np.roll(X, n_fwd_days+1, axis=0)

    X = shifted_X[n_fwd_days+1:]
    Y = Y[n_fwd_days+1:]

    n_time, n_stocks, n_factors = X.shape

    upper = np.nanpercentile(Y, upper_percentile, axis=1)[:, np.newaxis]
    lower = np.nanpercentile(Y, lower_percentile, axis=1)[:, np.newaxis]

    upper_mask = (Y >= upper)
    lower_mask = (Y <= lower)

    mask = upper_mask | lower_mask
    mask = mask.flatten()

    Y_binary = np.zeros(n_time * n_stocks)
    Y_binary[upper_mask.flatten()] = 1
    Y_binary[lower_mask.flatten()] = -1

    X = X.reshape((n_time * n_stocks, n_factors))

    X = X[mask]
    Y_binary = Y_binary[mask]

    return X, Y_binary

def get_last_values(input_data):
    last_values = []
    for dataset in input_data:
        last_values.append(dataset[-1])
    return np.vstack(last_values).T

class ML(CustomFactor):
    init = False

    def compute(self, today, assets, out, returns, *inputs):
        if (not self.init) or (today.weekday() == 0):
            self.imputer = preprocessing.Imputer()
            self.scaler = preprocessing.MinMaxScaler()
            log.info(today)
            self.clf = ensemble.AdaBoostClassifier(random_state=1337,
                                                   n_estimators=50)

            X = np.dstack(inputs)
            Y = returns

            X, Y = shift_mask_data(X, Y, n_fwd_days=PRED_N_FWD_DAYS)

            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)

            start = time()
            self.clf.fit(X, Y)
            end = time()
            log.info('Training took %f secs' % (end-start))

            self.init = True

        last_factor_values = get_last_values(inputs)
        last_factor_values = self.imputer.transform(last_factor_values)
        last_factor_values = self.scaler.transform(last_factor_values)


        out[:] = self.clf.predict_proba(last_factor_values)[:, 1]

def make_ml_pipeline(factors, universe, window_length=21, n_fwd_days=5):
    factors_pipe = OrderedDict()
    factors_pipe['Returns'] = Returns(inputs=[USEquityPricing.open],
                                      mask=universe, window_length=n_fwd_days + 1)

    for name, f in factors.iteritems():
        factors_pipe[name] = f().rank(mask=universe)

    factors_pipe['ML'] = ML(inputs=factors_pipe.values(),
                            window_length=window_length + 1,
                            mask=universe)


    pipe = Pipeline(screen=universe, columns=factors_pipe)

    return pipe

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    schedule_function(my_rebalance, TRADE_FREQ,
                      time_rules.market_open(minutes=1))

    schedule_function(my_record_vars, date_rules.every_day(),
                      time_rules.market_close())

    context.universe = Q1500US()
    ml_factors = make_factors()
    ml_pipeline = make_ml_pipeline(ml_factors,
                                   context.universe, n_fwd_days=PRED_N_FWD_DAYS,
                                   window_length=ML_TRAINING_WINDOW)
    attach_pipeline(ml_pipeline, 'alpha_model')

    context.past_predictions = {}
    context.hold_out_accuracy = 0
    context.hold_out_log_loss = 0
    context.hold_out_returns_spread_bps = 0

def evaluate_and_shift_hold_out(context):
    context.past_predictions = {k-1: v for k, v in context.past_predictions.iteritems() if k-1 >= 0}

    if 0 in context.past_predictions:
        returns = pipeline_output('alpha_model')['Returns'].rename('returns').to_frame()
        predictions = context.past_predictions[0].rename('predictions').to_frame()
        returns_predictions = returns.join(predictions, how='inner')
        returns_predictions['returns_binary'] = returns_predictions['returns'] > returns_predictions['returns'].median()
        returns_predictions['predictions_binary'] = returns_predictions['predictions'] > 0.5
        context.hold_out_accuracy = metrics.accuracy_score(returns_predictions['returns_binary'].values,
                                                           (returns_predictions['predictions_binary']).values)
        context.hold_out_log_loss = metrics.log_loss(returns_predictions['returns_binary'].values,
                                                     returns_predictions['predictions'].values)
        long_rets = returns_predictions.loc[returns_predictions.predictions_binary == 1, 'returns'].mean()
        short_rets = returns_predictions.loc[returns_predictions.predictions_binary == 0, 'returns'].mean()
        context.hold_out_returns_spread_bps = (long_rets - short_rets) * 10000

    context.past_predictions[PRED_N_FWD_DAYS] = context.predicted_probs

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.predicted_probs = pipeline_output('alpha_model')['ML']
    context.predicted_probs.index.rename(['date', 'equity'], inplace=True)


    evaluate_and_shift_hold_out(context)

    context.security_list = context.predicted_probs.index


def my_rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """

    predictions = context.predicted_probs

    predictions = predictions.loc[data.can_trade(predictions.index)]
    n_long_short = min(N_STOCKS_TO_TRADE // 2, len(predictions) // 2)
    predictions_top_bottom = pd.concat([predictions.nlargest(n_long_short),
                                        predictions.nsmallest(n_long_short)])

    predictions_top_bottom = predictions_top_bottom.iloc[~predictions_top_bottom.index.duplicated()]
    todays_universe = predictions_top_bottom.index

    predictions_top_bottom = (predictions_top_bottom - 0.5) * 2

    objective = opt.MaximizeAlpha(predictions_top_bottom)

    constrain_gross_leverage = opt.MaxGrossLeverage(1.0)
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(-.02, .02)
    market_neutral = opt.DollarNeutral()

    if predictions_top_bottom.index.duplicated().any():
        log.debug(predictions_top_bottom.head())
    order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
        ],
        universe=todays_universe,
    )


def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage,
           hold_out_accuracy=context.hold_out_accuracy,
           hold_out_log_loss=context.hold_out_log_loss,
           hold_out_returns_spread_bps=context.hold_out_returns_spread_bps,
    )

def handle_data(context,data):
    """
    Called every minute.
    """
    pass
