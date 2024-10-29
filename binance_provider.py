from datetime import datetime
from typing import List
import requests
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import logging

from common import AccountInfo, OrderType, Position, Side, Trade, TradeOrder, TransactionType, extract_asset_from_symbol, extract_stable_from_symbol, ExchangeDataProvider

load_dotenv()

logger = logging.getLogger(__name__)
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
# transactions = get_future_account_transaction_history(api_key, api_secret)

from binance.client import Client
import pandas as pd


def get_all_symbol_data(symbol, kline_size, client=None):
    client = client or Client("", "")
    start_ts = int(datetime.strptime('1 Jan 2017', '%d %b %Y').timestamp() * 1000)
    limit = 2500  # Adjust this value based on Binance's current limits
    all_klines = []

    while True:
        klines = client.get_historical_klines(symbol, kline_size, start_str=start_ts, limit=limit)
        if not klines:
            break

        all_klines.extend(klines)
        start_ts = klines[-1][0] + 1

    # Create a DataFrame
    df = pd.DataFrame(all_klines, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                           'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                                           'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])

    # Convert timestamps to datetime and adjust other data types as needed
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                                                                              'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']].astype(float)

    return df

def get_historical_data(symbol: str, interval: str, start_time: datetime, end_time: datetime, client=None) -> pd.DataFrame:
        """
        Fetch historical data for a given symbol and interval from Binance futures.
        Parameters:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        interval (str): The interval for kline/candlestick data (e.g., '1m', '1h', '1d').
        start_time (datetime): The start time for fetching data.
        end_time (datetime): The end time for fetching data.
        Returns:
        pd.DataFrame: A DataFrame containing historical kline/candlestick data with columns:
            - open_time (datetime): The open time of the kline.
            - open (float): The opening price.
            - high (float): The highest price.
            - low (float): The lowest price.
            - close (float): The closing price.
            - volume (float): The volume of the asset.
            - close_time (datetime): The close time of the kline.
            - quote_asset_volume (float): The volume of the quote asset.
            - number_of_trades (int): The number of trades.
            - taker_buy_base_asset_volume (float): The taker buy base asset volume.
            - taker_buy_quote_asset_volume (float): The taker buy quote asset volume.
        """
        client = client or Client("", "")
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)
        limit = 500  # Maximum number of data points per request
        
        historical_data = []
        while start_time_ms < end_time_ms:
            klines = client.futures_klines(
                symbol=symbol, 
                interval=interval, 
                startTime=start_time_ms, 
                endTime=end_time_ms, 
                limit=limit
            )

            if not klines:
                break

            for kline in klines:
                historical_data.append({
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'quote_asset_volume': float(kline[7]),
                    'number_of_trades': int(kline[8]),
                    'taker_buy_base_asset_volume': float(kline[9]),
                    'taker_buy_quote_asset_volume': float(kline[10])
                })

            # Move the start time forward based on the last returned kline's close_time
            start_time_ms = klines[-1][6] + 1  # Advance to the next millisecond after the last kline's close_time

        df = pd.DataFrame(historical_data)
        df.index = df['open_time']  
        return df


def fetch_and_save_all_usdt_pairs(client, kline_size, cg_api, ma=20):
    # Fetch exchange info
    info = client.get_exchange_info()
    symbols = info.get('symbols', [])

    # Filter for USDT pairs
    usdt_pairs = [s['symbol'] for s in symbols if s['symbol'].endswith('USDT')]

    dataset_folder = "./datasets/"
    os.makedirs(dataset_folder, exist_ok=True)
    cg_map = AssetMappings()

    # Use tqdm for progress display
    for symbol in tqdm(usdt_pairs, desc="Fetching USDT Pairs"):
        try:
            csv_filename = f"{dataset_folder}{symbol}_data.csv"
            if os.path.exists(csv_filename):
                print(f"Data for {symbol} already exists, skipping...")
                continue
            
            print(f"\nFetching data for {symbol}...")
            ticker = symbol.replace('USDT','')
            cg_id = cg_map.get_mapping(ticker)
            _,mcap,_ = asyncio.run( get_single_asset_data(cg_id,cg_api))
            df = get_all_symbol_data(symbol, kline_size, client)
            df.set_index('Timestamp', inplace=True)
            df = pd.merge_asof(df, mcap, left_index=True, right_index=True)
            df['ma'] = df['Close'].rolling(ma).mean()
            df['asset'] = ticker
            df.to_csv(csv_filename)
            print(f"Data for {symbol} saved to {csv_filename}")
            # Sleep for 60 seconds
            time.sleep(15)
        except Exception as e:
            print(f'Error processing {symbol}: {e}')
            pass
        
        
if __name__ == "__main__":

    client = Client(api_key, api_secret)
    kline_size = Client.KLINE_INTERVAL_1HOUR
