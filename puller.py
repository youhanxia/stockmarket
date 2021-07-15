import tushare as ts
import pandas as pd


def main():
    # 'sn', 'tt', 'nt'
    df = ts.get_tick_data('000725', date='2021-05-06', src='tt')
    df.to_csv('hist_data.csv', index=False)
    df = ts.get_today_ticks('000725')
    df.to_csv('today_data.csv', index=False)


if __name__ == '__main__':
    main()
