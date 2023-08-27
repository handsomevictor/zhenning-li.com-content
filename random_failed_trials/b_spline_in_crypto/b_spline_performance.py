import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


from random_failed_trials.b_spline_in_crypto.b_spline_model import get_prediction_result, get_processed_data


def simple_backtest(data, results_final, strategy=None):
    """
    确保data比results_final的长度多1
    """
    if len(data) != len(results_final) + 1:
        print("Length of data must be greater than length of results_final by 1.")
        raise ValueError

    # 如果strategy == 'simple'即预测的下一个点如果超过当前点，就买入，否则就卖出，每次买入卖出都是1个单位，假定手续费为0，且资金无限，
    # 且最终若position不等于0，可以选择继续持有或是按当前价卖出（这里假设先继续持有）
    if strategy == 'simple':
        # 初始化变量
        position = 0  # 当前仓位
        returns = []  # 收益列表
        positions = []  # 仓位随时间变化的列表
        wins = 0  # 胜利次数
        total_trades = 0  # 总交易次数
        winning_rates = []  # 胜率随时间变化的列表

        # 循环执行策略
        for i in range(len(results_final) - 1):  # -1是因为我们需要在下一期进行平仓
            if results_final[i] > data[i]:
                position += 1  # 买入 1 单位
            elif results_final[i] < data[i]:
                position -= 1  # 卖出 1 单位
            # 如果相等，什么都不做

            # 在下一期进行反向操作以平仓
            single_return = (data[i + 1] - data[i]) * position
            if position != 0:
                total_trades += 1
                if single_return > 0:
                    wins += 1
            returns.append(single_return)
            positions.append(position)

            # 计算并存储胜率
            winning_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            winning_rates.append(winning_rate)

            # 平仓
            position = 0
        # 绘图
        plt.figure(figsize=(12, 9))

        plt.subplot(3, 1, 1)
        plt.title('Cumulative Returns Over Time')
        plt.plot(np.cumsum(returns))
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')

        plt.subplot(3, 1, 2)
        plt.title('Position Over Time')
        plt.plot(positions)
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.subplot(3, 1, 3)
        plt.title('Winning Rate Over Time')
        plt.plot(winning_rates)
        plt.xlabel('Time')
        plt.ylabel('Winning Rate (%)')

        plt.tight_layout()
        plt.show()

        return returns, winning_rates[-1] if winning_rates else 0


if __name__ == '__main__':
    data = get_processed_data(minmax=True, data_type='ohlcvvwap', ohlcvvwap_interval='1m')
    print(data.shape)
    params = {
        "data": data,
        "window_size": 300,
        "num_prediction": 1,
        "degree": 2,
        "knots_density": 3,
        "plot": False
    }
    results_final = get_prediction_result(**params)

    # 将两者的长度对齐，原数据data长度比预测值多1
    data = np.array(data['price'].values[data.shape[0] - results_final.shape[1] - 1:])  # 这里-1只是因为当前点的实际值从这里开始
    results_final = np.array(results_final)[0]
    print(f'data shape is {data.shape}, results_final shape is {results_final.shape}')

    simple_backtest(data, results_final, strategy='simple')
