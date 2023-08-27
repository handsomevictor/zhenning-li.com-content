# from scipy.interpolate import BSpline
from scipy.interpolate import LSQUnivariateSpline  # 这个只能用于1维数据
# from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm


def get_processed_data(minmax=False, data_type='ohlcvvwap', ohlcvvwap_interval='1m'):
    if data_type == 'ohlcvvwap':
        original_data = pd.read_csv(os.path.join(os.getcwd(), 'crypto_data', 'binc_btc_usdt_ohlcvvwap.csv'),
                                    index_col=0)
        original_data.index = np.arange(1, len(original_data) + 1)
    elif data_type == 'trade':
        original_data = pd.read_csv(os.path.join(os.getcwd(), 'crypto_data', 'binc_btc_usdt_trade.csv'), index_col=0)
        # process the timestamp index - change to timestamp integer and make it start from 1
        original_data.index = pd.to_datetime(original_data.index, unit='ms')
        original_data.index = original_data.index.astype(int)
        original_data.index = original_data.index - original_data.index[0] + 1

    original_data['price'] = original_data['price'].fillna(method='ffill')
    original_data = original_data[['price']]

    # standardize the data
    if minmax == True:
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(original_data)
    elif minmax == False:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(original_data)
    elif minmax == 'Nothing':
        df_scaled = original_data.values
    df_scaled = pd.DataFrame(df_scaled, columns=original_data.columns, index=original_data.index)
    return df_scaled


def calculate_RMSE(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def rolling_window_prediction(data, window_size, num_prediction=1, degree=3, knots_density=10):
    # 从index=0开始，每次向后滚动window_size个单位，预测下一个单位的数据
    x = np.array(data.index)
    ys = [data[col].values for col in data.columns]  # 每个ys[i]代表数据的一个维度
    num_knots = int(window_size / knots_density)
    print(f"num_knots: {num_knots} in each window, window_size: {window_size}")
    results_final = []

    for i in tqdm(range(len(x) - window_size)):
        t = np.linspace(x[i], x[i + window_size], num_knots + 2)[1:-1].astype(int)
        current_x = x[i:i + window_size]
        current_ys = [y[i:i + window_size] for y in ys]
        # 为每个维度独立地创建B-spline
        splines = [LSQUnivariateSpline(current_x, current_y, t, k=degree) for current_y in current_ys]

        # 使用创建的splines为新数据点进行评估
        start_index = x[i]
        end_index = x[i + window_size]
        new_x = np.linspace(start_index, end_index + num_prediction, window_size + num_prediction)
        results = np.array([spline(new_x) for spline in splines])

        # only keep the last num_prediction results
        results = results[:, -num_prediction:]
        results_final.append(results)
    return np.array(results_final)


def get_prediction_result(data, window_size, num_prediction=1, degree=3, knots_density=8, plot=True):
    """
    画图，每一个维度一个图，每个图包含原始数据和预测数据
    """
    results_final = rolling_window_prediction(data=data,
                                              window_size=window_size,
                                              num_prediction=num_prediction,
                                              degree=degree,
                                              knots_density=knots_density)
    if num_prediction > 1:  # 这里并没有把bug解决
        result_final_temp = []
        for idx, vector in enumerate(results_final):
            if idx == 0:
                result_final_temp.append(results_final[idx][0][0])
            else:
                result_final_temp.append(np.mean([results_final[idx - 1][-1], vector[0]]))
        results_final = np.array(result_final_temp)
        print(f'shape of results_final is {results_final.shape}')
    else:
        # change the shape from (n, 8, 1) to (n, 8)
        results_final = results_final.reshape(results_final.shape[0], results_final.shape[1])
    results_final_index = np.arange(1 + window_size, len(results_final) + 1 + window_size)
    results_final = results_final.T

    x = np.array(data.index)
    ys = [data[col].values for col in data.columns]  # 每个ys[i]代表数据的一个维度

    # RMSE for each dimension
    RMSE_list = []
    for i in range(len(ys)):
        rmse_result = calculate_RMSE(ys[i][window_size:], results_final[i])
        print(f'RMSE for dimension {i + 1} is {rmse_result}')
        RMSE_list.append(rmse_result)
    print(f'Average RMSE is {np.mean(RMSE_list)}')

    if plot:
        # 绘制原始数据和预测数据
        for i in range(len(ys)):
            plt.figure(figsize=(10, 5))
            plt.plot(x, ys[i], 'o', label='Original Data')
            plt.plot(results_final_index, results_final[i], '-', label='Fitted B-spline')
            plt.legend()
            plt.xlabel('Time Index')
            plt.ylabel('Value')
            plt.title(f'Dimension {i + 1}')
            plt.show()
    return results_final


if __name__ == '__main__':
    data = get_processed_data(minmax=True, data_type='ohlcvvwap', ohlcvvwap_interval='1m')
    params = {
        "data": data,
        "window_size": 300,
        "num_prediction": 1,
        "degree": 2,
        "knots_density": 5,
        "plot": True
    }
    print(data.shape)
    results_final = get_prediction_result(**params)
    print(results_final.shape)
