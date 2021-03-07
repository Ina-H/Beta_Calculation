import datetime as dt
from pytickersymbols import PyTickerSymbols as PTS
import pandas as pd
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GetData:

    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        price_data = pdr.get_data_yahoo(self.ticker,
                                        start=self.start_date,
                                        end=self.end_date)['Adj Close']
        mtl_ret = price_data.resample('M').ffill().pct_change()[1:]

        return mtl_ret.values


class CalcBeta:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def linreg(self):
        x_sm = sm.add_constant(self.x)

        model = sm.OLS(self.y, x_sm).fit()
        # print(model.summary())

        return model.params[0], model.params[1]

    def covformula(self):
        variance = np.var(self.x)
        covariance = np.cov(self.y, self.x)[0][1]

        beta = covariance / variance

        return beta


if __name__ == "__main__":

    # end_date = dt.datetime.now()
    # start_date = dt.date(end_date.year - 5, end_date.month, end_date.day)
    end_date = '2020-12-31'
    start_date = '2015-11-30'

    ticker_index = 'DAX'  # SXXP für STOXX Europe 600
    # tickers = ['ADS','ALV','BAYN','DPW','DTE','HEI','LIN','MRK','MTX','SAP']
    stocks_germany = PTS().get_stocks_by_country('Germany')
    tickers = [item["symbol"] for item in stocks_germany]
    tickers = [t for t in tickers if t is not None]
    tickers.sort()
    # print(tickers)

    rf = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)[0].RF[1:]
    # print(rf)
    # rf = rf[1:]
    # rf.head()

    mtl_ret_index = GetData(ticker_index, start_date, end_date).download_data()
    mtl_ret_index = mtl_ret_index - rf.values
    # print(mtl_ret_index)

    data_alpha = []
    data_beta_linreg = []
    data_beta_covformula = []
    deviation = []
    tickers_done = []
    tickers_failed = []

    for t in tickers:
        try:
            mtl_ret_stock = GetData(t, start_date, end_date).download_data()
            mtl_ret_stock = mtl_ret_stock - rf.values
            X = mtl_ret_index
            Y = mtl_ret_stock

            alpha_linreg, beta_linreg = CalcBeta(X, Y).linreg()
            beta_cov = CalcBeta(X, Y).covformula()
            diff = beta_linreg - beta_cov

            # print('Alpha of ' + str(t) + ': ' + str(alpha_linreg))
            # print('Beta of ' + str(t) + ': ' + str(beta_linreg))
            data_alpha.append(alpha_linreg)
            data_beta_linreg.append(beta_linreg)
            data_beta_covformula.append(beta_cov)
            deviation.append(diff)
            tickers_done.append(t)

        except (RemoteDataError, ValueError, KeyError):
            tickers_failed.append(t)
            continue

    average_dev = sum([abs(value) for value in deviation])/len(deviation)

    data = {'Alpha via LinReg': data_alpha, 'Beta via LinReg': data_beta_linreg,
            'Beta via CovFormula': data_beta_covformula, 'Deviation': deviation}
    data_heading = ['Alpha via LinReg', 'Beta via LinReg', 'Beta via CovFormula', 'Deviation']
    df = pd.DataFrame(data, tickers_done, data_heading)
    # pd.set_option('precision',2)
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print(df)

    print('Average deviation when calculating beta via LinReg vs. CovFormula: ' + str(round(average_dev,2)))
    print('No data via Yahoo-Finance available: ' + str(tickers_failed))
