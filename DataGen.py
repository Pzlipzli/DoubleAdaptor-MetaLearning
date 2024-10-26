import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from Settings import *
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, window, return_window, start, end, freq, process):
        self.window = window
        self.return_window = return_window
        self.start = start
        self.end = end
        self.freq = freq
        self.path = path_dict[f'{freq}']
        self.process = process

        file_list = os.listdir(self.path)
        file_list.sort()
        self.latest_day = datetime.strptime(file_list[-1][:-4], '%Y-%m-%d') + pd.Timedelta(hours=8)

        self.df = self.__load_all_data()

        os.makedirs(path_dict['feature'], exist_ok=True)
        os.makedirs(path_dict['label'], exist_ok=True)

    def __load_all_data(self) -> pd.DataFrame:
        """
        :return: load minute data and daily data
        """
        start_date = (self.start - pd.Timedelta(days=(self.window - 1))).strftime('%Y-%m-%d')
        end_date = (self.end + pd.Timedelta(days=self.return_window + 1)).strftime('%Y-%m-%d')

        files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith('.csv')
                     and start_date <= file[:-4] <= end_date]

        df = self.__load_paths(files, freq=self.freq)

        return df

    def __load_paths(self, path_list: list, freq: str) -> pd.DataFrame:
        """
        :param path_list: a list of paths
        :return: a dataframe of all the data (volume is log transformed)
        """
        dfs_all = []
        path_list.sort()

        for file_path in path_list:
            df = pd.read_csv(file_path)
            df = df[['开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '主动买入成交量', '货币对']]
            df.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'positive_vol', 'token']
            df['openTime'] = pd.to_datetime(df['openTime'])
            df.sort_values(by=['token', 'openTime'], inplace=True)

            start_dt = datetime.strptime(file_path[-14:-4], '%Y-%m-%d') + pd.Timedelta(hours=8)\
                if freq == self.freq else datetime.strptime(file_path[-14:-4], '%Y-%m-%d')

            df_empty = self.get_empty_df(start_dt, freq)
            df = df.groupby('token').apply(self.empty_process, empty=df_empty)
            df.reset_index(drop=True, inplace=True)

            df['volume'] = np.log(df['volume'] + 0.1)
            df['positive_vol'] = np.log(df['positive_vol'] + 0.1)
            dfs_all.append(df)

        data = pd.concat(dfs_all, axis=0)
        data.sort_values(by=['token', 'openTime'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

    @staticmethod
    def empty_process(group, empty):
        """
        front-fill empty df
        :param group: groupby token
        :param empty: empty df
        :return: front-filled df
        """
        group.reset_index(drop=True, inplace=True)
        df_merge = pd.merge(empty, group, how='left', left_on='openTime', right_on='openTime')
        return df_merge.ffill()

    @staticmethod
    def get_empty_df(start: datetime, frequency: str):
        """
        create empty df, used for front-fill
        :param start: start date
        :param frequency: frequency
        : return: empty df
        """
        open_list = pd.date_range(start=start, end=start + pd.Timedelta(days=1), freq=frequency, inclusive="left")

        df_empty = pd.DataFrame({"openTime": open_list})
        df_empty["closeTime"] = df_empty["openTime"] + pd.Timedelta(frequency) - pd.Timedelta(seconds=1)

        return df_empty

    def run(self):
        dates = pd.date_range(start=self.start, end=self.end).tolist()  # 日期列表
        dates = [dt + pd.Timedelta(hours=8) for dt in dates]

        for dt in tqdm(dates):
            data = self.df[((dt - pd.Timedelta(days=self.window - 1)) <= self.df['openTime'])
                                   & (self.df['closeTime'] <= (dt + pd.Timedelta(days=self.return_window + 2)))]
            future_data = data[['openTime', 'closeTime', 'token', 'close']].copy()

            data = data[data['closeTime'] <= (dt + pd.Timedelta(days=1))].reset_index(drop=True)
            future_data = future_data[future_data['openTime'] >= dt].reset_index(drop=True)

            past_token = data[data['openTime'] == (dt - pd.Timedelta(days=self.window - 1))].token.unique()
            now_token = future_data[future_data['openTime'] == (dt + pd.Timedelta(days=1))].token.unique()

            if len(now_token) == 0:
                now_token = future_data[future_data['openTime'] == (dt + pd.Timedelta(days=0))].token.unique()

            tokens = list(set(now_token) & set(past_token))

            with ProcessPoolExecutor(max_workers=self.process) as executor:
                get_with_args = partial(self.get_one_token, data=data, future_data=future_data)
                futures = [executor.submit(get_with_args, token) for token in tokens]
                results = [future.result() for future in as_completed(futures)]

            data_lists = [result[0] for result in results]
            future_lists = [result[1] for result in results]

            df_data = pd.concat(data_lists, ignore_index=True)
            df_data['openTime'] = dt
            df_future = pd.concat(future_lists, ignore_index=True)
            df_data.sort_values(by=['token'], inplace=True)
            df_future.sort_values(by=['token', 'openTime'], inplace=True)

            df_data.to_csv(path_dict['feature'] + f'{dt.strftime("%Y-%m-%d")}.csv', index=False)

            if dt <= (self.latest_day - pd.Timedelta(days=self.return_window)):
                df_future.to_csv(path_dict['label'] + f'{dt.strftime("%Y-%m-%d")}.csv', index=False)

            print(f'{dt.strftime("%Y-%m-%d")} is saved.')

    def get_one_token(self, token, data, future_data):
        data_token = data[data['token'] == token].reset_index(drop=True)
        dt = data_token['openTime'].iloc[-1]

        empty_feature = pd.DataFrame()
        empty_label = pd.DataFrame()
        empty_feature['token'] = token
        empty_label['token'] = token
        empty_feature['openTime'] = dt
        empty_label['openTime'] = dt

        if len(data_token) < self.window:
            return empty_feature, empty_label

        last_close = data_token.iloc[-1, 5]
        data_token[['open', 'close', 'high', 'low']] = data_token[['open', 'close', 'high', 'low']] / last_close

        melted = pd.melt(data_token, id_vars=['openTime'],
                         value_vars=['open', 'high', 'low', 'close', 'volume', 'positive_vol'])
        melted.set_index(['openTime', 'variable'], inplace=True)
        melted = melted.T
        melted.reset_index(inplace=True, drop=True)

        # rename columns
        melted.columns = [f"{s}{i}" for s in ['open', 'high', 'low', 'close', 'volume', 'positive_vol']
                          for i in range(-59, 1)]
        melted['token'] = token

        future_data_token = future_data[future_data['token'] == token].reset_index(drop=True)

        if len(future_data_token) < self.return_window + 2:
            if dt + pd.Timedelta(days=self.return_window) >= self.latest_day:
                return melted, empty_label
            else:
                return empty_feature, empty_label
        future_data_token['openTime'] = future_data_token['openTime'].shift(1)
        future_data_token['closeTime'] = future_data_token['closeTime'].shift(1)
        future_data_token['future_close'] = future_data_token['close'].shift(-self.return_window)
        future_data_token['future_return'] = future_data_token['future_close'] / future_data_token['close'] - 1
        future_data_token.drop(columns=['close', 'future_close'], inplace=True)
        future_data_token.dropna(inplace=True)

        return melted, future_data_token


def generate_date_ranges(start_date=None, end_date=None, mode='generate', freq='1D'):
    file_list = sorted(os.listdir(path_dict[freq]))
    if mode == 'generate':
        assert start_date and end_date, 'start_date and end_date must be provided in generate mode'
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    elif mode == 'update' and file_list:  # update mode, with existent data
        start_date = pd.to_datetime(file_list[-1][:-4]) + pd.Timedelta(days=1)
        end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d 00:00:00')) - pd.Timedelta(days=1)
    else:
        raise ValueError("'update' is not for first time running, try 'generate' instead")

    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + pd.DateOffset(months=2) - pd.Timedelta(days=1)

        if current_end > end_date:
            current_end = end_date

        yield current_start, current_end

        current_start = current_end + pd.Timedelta(days=1)


if __name__ == '__main__':
    window = 60
    return_window = 7
    start = pd.to_datetime('2024-08-28')
    end = pd.to_datetime('2024-09-08')
    freq = '1D'
    process = 4
    mode = 'generate'  # choose the mode: generate and update

    for start, end in tqdm(generate_date_ranges(start, end, mode)):
        dataloader = DataLoader(window, return_window, start, end, freq, process)
        dataloader.run()
        del dataloader
