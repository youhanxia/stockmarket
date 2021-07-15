import os
import datetime
import pickle
import numpy as np
import pandas as pd
import tushare as ts

from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

year, month = '2021', '04'

global_selected = [
    '000725',
    '002714',
    '300059',
    '510300',
    '510500',
    '600036',
    '600867',
    '600999',
]

type_encoder = {
    '卖盘': 1,
    '中性盘': 0,
    '买盘': -1,
}

data_dir = 'data'


def get_daily_file(selected=None):
    if selected is None:
        selected = global_selected

    for code in selected:
        df = pd.DataFrame()
        for date in (datetime.date(int(year), int(month), 1) + datetime.timedelta(i) for i in range(30)):
            df_ = ts.get_tick_data(code, date=str(date), src='tt')
            # df_ = ts.get_today_ticks(code)
            if df_ is None:
                continue

            df_['date'] = date
            df = df.append(df_[['date', 'time', 'price', 'volume', 'type']])

        df['type'].replace(type_encoder, inplace=True)

        df.to_csv(os.path.join(data_dir, ''.join([code, '_', year, '-', month, '.csv'])), index=False)


def time_series_prep(selected=None, timestep=600, samplestep=10, window=12):
    num_series = timestep // samplestep
    dates = None

    if selected is None:
        selected = global_selected
    for code in selected:
        dn = ''.join([code, '_', year, '-', month])
        df = pd.read_csv(os.path.join(data_dir, dn + '.csv'))

        enc_in = []
        dec_in = []
        output = []

        if dates is None:
            dates = list(df['date'].drop_duplicates())

        for date in dates:
            print('\r', code, date, end='')
            time_bucket = [[] for _ in range(num_series)]
            df_ = df[(df['date'] == date) & (df['time'] >= '09:30:00')]
            for i, row in df_.iterrows():
                time = [int(t) for t in row['time'].split(':')]
                time = time[0] * 3600 + time[1] * 60 + time[2]
                time_delta = time - 34200         # 9:30
                time_delta //= samplestep
                length, index = time_delta // num_series, time_delta % num_series
                while len(time_bucket[index]) < length:
                    time_bucket[index].append(row[['price', 'volume', 'type']].to_list())

            for series in time_bucket:
                series = np.array(series)
                for i in range(len(series) - 23):
                    enc_in.append(series[i: i + window])
                    dec_in.append(series[i + window - 1: i + 2 * window - 1, 0])
                    output.append(series[i + window: i + 2 * window, 0])

        enc_in = np.array(enc_in)
        dec_in = np.array(dec_in)
        output = np.array(output)

        with open(os.path.join(data_dir, dn + '.pkl'), 'wb') as f:
            pickle.dump({
                'enc_in': enc_in,
                'dec_in': dec_in,
                'output': output,
            }, file=f)


if __name__ == '__main__':
    get_daily_file(['510300', '510500'])
    time_series_prep(['510300', '510500'])

    os.system('say "Mission complete."')
