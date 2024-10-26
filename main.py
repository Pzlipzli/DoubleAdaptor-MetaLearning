import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
import os
from datetime import datetime, timedelta
import random
import copy

from DataAdaptor import DataTransformer
from LSTM import LSTMModel
from utils import get_tokens, get_dates, cal_corr
from Settings import path_dict


class Trainer:
    def __init__(self, save_path, feature_path, label_path, factor_path, num_trans=8, num_days=40, split_day=20,
                 trans_path='no_model', maml_path='no_model'):
        self.num_days = num_days
        self.split_day = split_day
        self.num_trans = num_trans

        self.save_path = save_path
        self.feature_path = feature_path
        self.label_path = label_path
        self.factor_path = factor_path

        self.trans_path = trans_path
        self.maml_path = maml_path

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(factor_path, exist_ok=True)

    def get_tasks(self, start_dates):
        """
        Generate a lis containing all tasks according to the start dates of each task
        :param start_dates: a list of start dates of each task
        :return: a list containing related tasks
        """
        tasks = []

        for date in start_dates:
            x_train, x_valid = self.load_and_process_data(self.feature_path, date)
            y_train, y_valid = self.load_and_process_data(self.label_path, date)
            tasks.append((x_train, x_valid, y_train, y_valid))

        return tasks

    def load_and_process_data(self, path, start_date):
        """
        Load csv files according to the start date and the length of period.
        Filter out the assets without enough data and make the rest into two tensors.
        :param path: the folder path of csv files
        :param start_date: start date（format: 'YYYY-MM-DD'）
        :return: two tensors--support set and query set
            support set: a tensor of the first split_day-th days' data，
            with the shape of (batch_size, split_day, feature_count)
            query set: a tensor of the last (num_days-split_day)-th days' data，
            with the shape of (batch_size, num_days-split_day, feature_count)
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=self.num_days)

        dataframes = []
        temp = os.listdir(path)
        temp.sort()

        # load and concat files
        for file in os.listdir(path):
            try:
                file_date = datetime.strptime(file.split('.')[0], '%Y-%m-%d')
            except ValueError:
                continue

            # focus on needed dates' files
            if start_date <= file_date < end_date:
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                dataframes.append(df)

        if not dataframes:
            raise ValueError("No data files found within the specified date range.")

        all_data = pd.concat(dataframes, ignore_index=True)

        all_data['openTime'] = pd.to_datetime(all_data['openTime'])
        all_data = all_data.sort_values(by=['token', 'openTime'])

        try:
            all_data = all_data.drop(columns=['closeTime'])
        except:
            pass

        assets = all_data['token'].unique()
        assets.sort()

        first_days_data = []
        second_days_data = []

        for asset in assets:
            asset_data = all_data[all_data['token'] == asset].drop(columns=['token']).set_index('openTime')

            first_part = asset_data[start_date:start_date + timedelta(days=self.split_day)].values
            second_part = asset_data[start_date + timedelta(days=self.split_day):start_date +
                                                                                 timedelta(days=self.num_days)].values
            # filter condition: with enough data
            if len(first_part) == self.split_day:
                first_days_data.append(first_part)
            if len(second_part) == self.num_days - self.split_day:
                second_days_data.append(second_part)

        if not first_days_data:
            raise ValueError("No assets found with sufficient data in the specified date range.")

        # turn into tensors with shape of (time_steps, num_assets, features)
        try:
            first_tensor = torch.tensor(np.array(first_days_data), dtype=torch.float32).permute(1, 0, 2)
        except:
            first_tensor = torch.tensor(np.array(first_days_data), dtype=torch.float32)

        try:
            second_tensor = torch.tensor(np.array(second_days_data), dtype=torch.float32).permute(1, 0, 2)
        except:
            second_tensor = torch.tensor(np.array(second_days_data), dtype=torch.float32)

        return first_tensor, second_tensor

    def maml_training_loop(self, tasks, num_iterations=200, meta_lr=0.01, trans_lr=0.01, patience=4, mode='train'):
        """
        Training function: train on given tasks.
        Early stop in .
        Save the model.
        """
        if mode == 'train':
            trains = tasks[0]
            valids = tasks[1]
        else:
            trains = tasks
            valids = []

        # Define the meta-optimizer
        meta_optimizer = optim.Adam(self.maml.parameters(), lr=meta_lr)
        trans_optimizer = optim.Adam(self.data_transformer.parameters(), lr=trans_lr)

        # Loss function
        loss_fn = nn.MSELoss()

        last_ic = -float('inf')
        epochs_decrease = 0
        last_learner_state = None
        last_transformer_state = None

        for iteration in range(num_iterations):

            for task_data in trains:
                learner = self.maml.clone()

                meta_optimizer.zero_grad()
                trans_optimizer.zero_grad()

                total_loss, _ = self.train_step(task_data, learner, loss_fn)
                total_loss.backward()

                meta_optimizer.step()
                trans_optimizer.step()

            if mode != 'train':  # predict mode
                continue

            # Keep the original model
            saved_maml = self.maml.clone()
            saved_trans = copy.deepcopy(self.data_transformer)
            total_ic = 0

            for task_data in valids:
                learner = self.maml.clone()

                meta_optimizer.zero_grad()
                trans_optimizer.zero_grad()

                total_loss, mean_ic = self.train_step(task_data, learner, loss_fn)
                total_loss.backward()
                total_ic += mean_ic

                meta_optimizer.step()
                trans_optimizer.step()

            # Restore the original model
            self.maml = saved_maml
            self.data_transformer = saved_trans

            # Early stopping logic-- IC keeps going down for consecutive patience times
            if total_ic > last_ic:
                epochs_decrease = 0
                last_learner_state = self.maml.state_dict()
                last_transformer_state = self.data_transformer.state_dict()
            else:
                epochs_decrease += 1
            last_ic = total_ic

            # Check if we should stop early
            if epochs_decrease >= patience:
                print(f"Early stopping at iteration {iteration + 1}")
                break

            print(f"Iteration {iteration + 1}: Meta-valid task cumulative ICIR: {total_ic:.6f},"
                  f" decrease count: {epochs_decrease}")

        # Save final model and data transformer state
        if self.save_path:
            if last_learner_state is not None and last_transformer_state is not None and mode == 'train':
                torch.save(last_learner_state, self.save_path + "trained_maml_model.pth")
                torch.save(last_transformer_state, self.save_path + "trained_data_transformer.pth")
                print("Trained model saved successfully.")
            elif mode != 'train':
                torch.save(self.maml.state_dict(), self.save_path + "trained_maml_model.pth")
                torch.save(self.data_transformer.state_dict(), self.save_path + "trained_data_transformer.pth")
                print("Predict model saved successfully.")
            else:
                print("Warning: No model was saved because early stopping occurred before the first iteration.")

    def train_step(self, task_data, learner, loss_fn, alpha=0.2):
        """
        Train on one task
        """
        # Split task data into support and query sets
        support_data, query_data, support_labels, query_labels = task_data

        support_data = self.data_transformer(support_data)
        support_labels = self.data_transformer(support_labels, is_y=True)
        query_data = self.data_transformer(query_data)

        # Adapt the model to the task
        support_preds = learner(support_data)
        support_loss = loss_fn(support_preds, support_labels)

        learner.adapt(support_loss)

        # Compute the meta-gradient
        query_preds = learner(query_data)
        query_preds = self.data_transformer.inverse_transform(query_preds)
        query_loss = loss_fn(query_preds, query_labels)

        _ = self.data_transformer(support_data)
        reg_data = self.data_transformer(support_labels, is_y=True)
        reg_loss = loss_fn(reg_data, support_labels)

        total_loss = query_loss + alpha * reg_loss

        mean_ic = cal_corr(query_preds, query_labels)

        return total_loss, mean_ic

    def online_perdict(self, task, dates, days=20):
        """
        Predict on given task and save the factors
        """
        _, x_test, _, _ = task

        with torch.no_grad():
            x_test = self.data_transformer(x_test)
            y_pred = self.maml(x_test)
            y_pred = self.data_transformer.inverse_transform(y_pred).squeeze(-1).permute(1, 0)

        assets = get_tokens(dates[0], self.feature_path, days=days)
        assets.sort()

        for j, dt in enumerate(dates):
            list_one = []
            for i, asset in enumerate(assets):
                y_one = y_pred[i:i+1, j:j+1]

                df = pd.DataFrame(y_one.numpy(), columns=['上涨概率'])
                df['证券代码'] = asset
                df['日期'] = dt

                list_one.append(df)

            df_one = pd.concat(list_one)
            if os.path.exists(self.factor_path + f'{dt.strftime("%Y-%m-%d")}.csv'):
                print(f'Factor {dt} skipped.')
            else:
                df_one.to_csv(self.factor_path + f'{dt.strftime("%Y-%m-%d")}.csv', index=False)
                print(f'Factor {dt} saved.')

    def run(self, tasks, dates, iter=200, task_lr=0.01, mode='train', divide=0):
        """
        Run according to the mode
        """
        if not tasks:
            return

        x_train, x_valid, y_train, y_valid = tasks[0] if mode != 'train' else tasks[0][0]
        num_features = x_train.shape[-1]
        output_size = y_train.shape[-1]

        self.data_transformer = DataTransformer(num_features, self.num_trans)  # Create the data-adaptor class

        # Create LSTM model (GRU) and wrap with MAML
        model = LSTMModel(input_size=num_features, hidden_size=128, num_layers=2, output_size=output_size)
        self.maml = l2l.algorithms.MAML(model, lr=task_lr)

        print(f'{mode} mode starts.')

        if mode == 'predict':
            for i, task in enumerate(tasks):
                if self.maml_path and self.trans_path:
                    self.data_transformer.load_state_dict(
                        torch.load(self.save_path + self.trans_path, weights_only=True))
                    self.maml.load_state_dict(torch.load(self.save_path + self.maml_path, weights_only=True))
                else:
                    raise ValueError('There is no model to load')

                self.data_transformer.train()
                self.maml.train()
                if len(task[3]) == 0:
                    print('No enough data for training.')
                    break

                self.maml_training_loop(tasks=[task], num_iterations=iter, mode='predict', patience=2)

                self.data_transformer.eval()
                self.maml.eval()

                if i >= divide - 1:
                    date = datetime.strptime(dates[i], '%Y-%m-%d')
                    date_list = [date + pd.Timedelta(days=i + 40) + pd.Timedelta(hours=8) for i in range(20)]
                    try:
                        self.online_perdict(tasks[i + 1], date_list)
                    except:
                        print('No enough data for predicting.')
                        pass

        elif mode == 'update':  # For the latest task
            if self.maml_path and self.trans_path:
                self.data_transformer.load_state_dict(torch.load(self.save_path + self.trans_path, weights_only=True))
                self.maml.load_state_dict(torch.load(self.save_path + self.maml_path, weights_only=True))

            for date in dates:
                if os.path.exists(self.feature_path + date + '.csv'):
                    df = pd.read_csv(self.feature_path + date + '.csv')
                    assets = df['token'].unique()
                    assets.sort()

                    tensor_data = []

                    for asset in assets:
                        asset_data = df[df['token'] == asset].drop(columns=['token']).set_index('openTime')
                        tensor_data.append(asset_data)

                    try:
                        tensor_data = torch.tensor(np.array(tensor_data), dtype=torch.float32).permute(1, 0, 2)
                    except:
                        tensor_data = torch.tensor(np.array(tensor_data), dtype=torch.float32)
                    task = [0, tensor_data, 0, 0]

                    self.online_perdict(task, dates=[datetime.strptime(date, '%Y-%m-%d')], days=1)

        else:
            self.maml_training_loop(tasks=tasks, num_iterations=iter, mode=mode)

        print(f'{mode} mode ends')

    def main(self, start_date, train_end, valid_end, test_end):
        """
        Train on the train set and generate factors for the consecutive 360 days
        """
        train_dates, valid_dates, test_dates, update_dates = get_dates(start_date, train_end, valid_end, test_end, label)

        train_tasks = self.get_tasks(train_dates)
        random.shuffle(train_tasks)
        valid_tasks = self.get_tasks(valid_dates)
        whole_tasks = [train_tasks, valid_tasks]
        self.run(whole_tasks, train_dates, iter=200, mode='train')

        test_tasks = self.get_tasks(test_dates)
        whole_test = valid_tasks + test_tasks
        whole_dates = valid_dates + test_dates
        self.run(whole_test, whole_dates, iter=30, mode='predict', divide=len(valid_tasks))

        if not test_tasks:
            test_tasks = self.get_tasks(train_dates)
        update_tasks = test_tasks
        self.run(update_tasks, update_dates, iter=1, mode='update')


if __name__ == '__main__':
    feature = path_dict['feature']
    label = path_dict['label']
    save_path = path_dict['save_path']
    factor = path_dict['factor']

    start_date = '2018-01-01'
    train_end = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=1080)).strftime('%Y-%m-%d')
    valid_end = (datetime.strptime(train_end, '%Y-%m-%d') + timedelta(days=360)).strftime('%Y-%m-%d')
    end_date = '2024-09-08'
    test_end = valid_end

    trainer = Trainer(save_path, feature, label, factor, trans_path='trained_data_transformer.pth',
                      maml_path='trained_maml_model.pth')

    # Every 360 days, we redo the whole process of training and generating factors
    while True:
        test_end = (datetime.strptime(valid_end, '%Y-%m-%d') + timedelta(days=360)).strftime('%Y-%m-%d')
        if test_end > end_date:
            break
        trainer.main(start_date, train_end, valid_end, test_end)
        train_end, valid_end = valid_end, test_end

    trainer.main(start_date, train_end, valid_end, end_date)

