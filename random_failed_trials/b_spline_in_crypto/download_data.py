import requests
import pandas as pd
import os

api_key = os.environ['KAIKO_API_KEY']


def main(end_point):
    if end_point == f'trade':
        url = ('https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/binc/spot/btc-usdt/trades'
               '?end_time=2023-08-27T21:00:00.000Z'
               '&start_time=2023-08-25T00:00:00.000Z'
               '&page_size=100000')
    elif end_point == 'ohlcvvwap':
        url = ('https://us.market-api.kaiko.io/v1/data/trades.v1/exchanges/binc/spot/btc-usdt/aggregations'
               '/count_ohlcv_vwap'
               '?start_time=2023-08-25T00:00:00.000Z'
               '&end_time=2023-08-27T21:00:00.000Z'
               '&interval=1m'
               '&page_size=100000')

    headers = {"Accept": "application/json", "X-Api-Key": api_key}
    response = requests.get(url, headers=headers).json()
    res = pd.DataFrame(response['data'])
    page = 1
    while "next_url" in response:
        print(page)
        response = requests.get(response['next_url'], headers=headers).json()
        res = pd.concat([res, pd.DataFrame(response['data'])], axis=0)
        page += 1

    res['timestamp'] = pd.to_datetime(res['timestamp'], unit='ms')
    return res


if __name__ == '__main__':
    end_point = 'ohlcvvwap'
    res = main(end_point)
    res.to_csv(os.path.join(os.getcwd(), 'crypto_data', 'binc_btc_usdt_ohlcvvwap.csv'), index=False)
