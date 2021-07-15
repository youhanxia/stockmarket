import os
import datetime
import pickle
import numpy as np
import pandas as pd
import tushare as ts


def calc_cyc_cys(selected, period=13):
    cyc = []
    cur_prices = []
    for code in selected:
        df = ts.get_today_ticks(code)
        df.rename(columns={'vol': 'volume'}, inplace=True)
        df = df[['price', 'volume']]
        i = int(df is not None)
        if i:
            cur_prices.append(np.array(df['price'])[-1])
        else:
            df = pd.DataFrame()
        date = datetime.date.today()
        while i < period:
            date += datetime.timedelta(-1)
            df_ = ts.get_tick_data(code, date=str(date), src='tt')
            if df_ is None:
                continue
            df.append(df_[['price', 'volume']])
            if not i:
                cur_prices.append(np.array(df['price'])[-1])
            i += 1
        df['amount'] = df['price'] * df['volume']
        cyc.append(df['amount'].sum() / df['volume'].sum())
    cys = [p / c - 1 for p, c in zip(cur_prices, cyc)]
    return cyc, cys


if __name__ == '__main__':
    print(calc_cyc_cys(['510300', '510500']))

