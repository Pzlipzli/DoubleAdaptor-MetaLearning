from datetime import datetime, timedelta
import pandas as pd
import os


def get_tokens(start_date, folder_path, days=20):
    # Generate a list of file names of `days`
    file_names = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d.csv') for i in range(days)]

    name_sets = []

    for file_name in file_names:
        if os.path.exists(folder_path + file_name):
            df = pd.read_csv(folder_path + file_name)
            df.sort_values(by='token', inplace=True)
            name_sets.append(set(df['token']))
        else:
            print(f"File {file_name} not found.")

    if name_sets:
        common_names = set.intersection(*name_sets)
        return list(common_names)
    return []


def find_date_ranges(start_date_str, directory, days=20):
    """
    Check whether the consecutive files exist according to the start_date
    Return a list of the consecutive dates
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    files = os.listdir(directory)
    date_files = [f for f in files if f.endswith('.csv')]

    date_set = set()
    for file in date_files:
        try:
            file_date = datetime.strptime(file.split('.')[0], '%Y-%m-%d')
            date_set.add(file_date)
        except ValueError:
            continue

    valid_dates = []
    current_date = start_date

    while True:
        all_dates_exist = True
        for i in range(days):
            check_date = current_date + timedelta(days=i)
            if check_date not in date_set:
                all_dates_exist = False
                print(f"日期 {check_date.strftime('%Y-%m-%d')} 不存在")
                break

        if all_dates_exist:
            valid_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=days)
        else:
            break

    return valid_dates


def get_dates(start_date, train_end, valid_end, end_date, folder):
    start_dates = find_date_ranges(start_date, folder)
    train_dates = [date for date in start_dates if datetime.strptime(date, '%Y-%m-%d')
                   <= datetime.strptime(train_end, '%Y-%m-%d')]
    valid_dates = [date for date in start_dates if date not in train_dates and
                  datetime.strptime(date, '%Y-%m-%d') <= datetime.strptime(valid_end, '%Y-%m-%d')]
    test_dates = [date for date in start_dates if (date not in train_dates) and (date not in valid_dates) and
                  datetime.strptime(date, '%Y-%m-%d') <= datetime.strptime(end_date, '%Y-%m-%d')]

    last_start_date = datetime.strptime(start_dates[-1], '%Y-%m-%d')
    start_update_date = last_start_date + timedelta(days=20)

    current_date = start_update_date
    update_dates = []
    while current_date <= datetime.strptime(end_date, '%Y-%m-%d'):
        update_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return train_dates, valid_dates, test_dates, update_dates

def cal_corr(tensor1, tensor2):
    """
    Calculate IC (Rank-IC, IR)
    You can change according to your demand
    :param tensor1: t*n*1 tensor
    :param tensor2: t*n*1 tensor
    :return: mean ic
    """
    tensor1 = tensor1.squeeze(-1)
    tensor2 = tensor2.squeeze(-1)
    n = tensor1.size(1)

    mean_tensor1 = tensor1.mean(dim=1, keepdim=True)
    mean_tensor2 = tensor2.mean(dim=1, keepdim=True)

    diff_tensor1 = tensor1 - mean_tensor1
    diff_tensor2 = tensor2 - mean_tensor2

    covariance = (diff_tensor1 * diff_tensor2).sum(dim=1) / (tensor1.size(1) - 1)

    std_tensor1 = diff_tensor1.std(dim=1, unbiased=True)
    std_tensor2 = diff_tensor2.std(dim=1, unbiased=True)

    correlation_coefficients = covariance / (std_tensor1 * std_tensor2) / n

    average_correlation = correlation_coefficients.mean()

    return average_correlation

