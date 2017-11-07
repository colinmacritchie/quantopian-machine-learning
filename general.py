from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import Latest, CustomFactor, SimpleMovingAverage, AverageDollarVolume, Returns, RSI, RollingLinearRegressionOfReturns
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.filters import Q500US, Q1500US
from quantopian.pipeline.data.quandl import fred_usdontd156n as libor

import quantipian.experimental.optimize as opt

import talib
import pandas as pd
import numpy as np
from time import time
from collections import OrderedDict

from scipy import stats
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics, svm, cross_validation

# Config of global strategy

N_STOCKS_TO_TRADE = 500 #Split 50% and 50% short - Takes a random set of 500 companies.
ML_TRADING_WINDOW = 125 #Number of days to train the alg to work with before acting.
PRED_N_FWD_DAYS = 1
TRADE_FREQ = date_rules.everyday()

#Pipeline Factors

bs = morningstar.balance_sheet
cfs = morningstar.cash_flow_statement
is_ = morningstar.inmcome_statement
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
            (cfs.free_cash_flow.latest *4.)

    def EBITDA_Yield():
        return (_is.ebita.latest * 4.) / \
            USEquityPricing.close.latest

    def EBIT_To_Assets():
        return (is_.ebit.latest * 4.) / \
           bs.total_assets.latest

    def Return_On_Total_Investment_Capital():
        return or_.roic.latest

    class Mean_Reversion_1m(CustomFactor):
        inputs = [Returns(window_length=21)
        window_length = 252

       def compute(self, today, assets, out, monthly_rets):
           out[:] = (monthly_rets[-1] - np.nanmean(monthly_rets, axis=0)) / \
               np.natnstd(monthly_rets, axis=0)

    class MACD_Signal_10d(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 60

        def compute(self, today, assets, out, close):
            
            sig_lines = []
	    
	    for col in close.T:
	        try:
                   _, signal_line, _ = tablib.MACD(col, fasterperiod=12,
						   slowperiod=26, signalperiod=10)
                   sig_lines.append(signal_line[-1])
                except:
                    sig_lines.append(np.nan)
             out[:] = sig_lines

     class MoneyFlow_Volume_5d(CustomFactor):
         inputs = [UsEquityPricing.close, USEquityPricing.volume]
         window_length = 5

         def compute(self, today, assets, out, close, volume):
             mfvs = []
             for col_c, col_v, in zip(close.T, volume.T):
                 denominator = np.dot(col_c, col_v)
                 numerator = 0.
                 for n, price in enumerate(col_c.tolist())
	             if price > col_c[n - 1]:
			numerator += price * col_v[n]
		     else:
			numerator -= price * col_v[n]
		 mfvs.append(numerator / denominator)
              out[:] = mfvs

      def Net_Income_margin():
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
 	      X_matrix = np.title(X_vector, (len(close.T), 1)).T

              Y_bar = np.nanmean(close, axis=0)
	      Y_bars = np.title(Y_bar, (self.window_length, 1))
	      Y_matrix = close - Y_bars

	      x_var = np.nanvar(X)


              out[:] = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / \
		  (self.window_length)

      
         
