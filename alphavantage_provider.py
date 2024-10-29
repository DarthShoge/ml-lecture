import requests
import pandas as pd
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
base_url = 'https://www.alphavantage.co/query'
max_retries = 5  # Number of retries in case of failure

def get_time_series_data_for_month(function, symbol, interval, month, output_size='full', adjusted='true', extended_hours='true', retries=0):
    """Fetches time series data for a specific month from Alpha Vantage and returns it as a pandas DataFrame.
    
    Args:
        function (str): Alpha Vantage function (e.g., 'TIME_SERIES_INTRADAY').
        symbol (str): The stock ticker (e.g., 'IBM').
        interval (str): Data interval (e.g., '5min', '15min'). Required for intraday functions.
        month (str): The month to fetch data for in 'YYYY-MM' format.
        output_size (str): Output size ('compact' or 'full'). Defaults to 'full'.
        adjusted (str): 'true' or 'false' for adjusted data. Defaults to 'true'.
        extended_hours (str): 'true' or 'false' to include extended hours. Defaults to 'true'.
        retries (int): Number of retries if the request fails.

    Returns:
        pd.DataFrame: Time series data as a DataFrame or None if request fails.
    """
    params = {
        'function': function,
        'symbol': symbol,
        'interval': interval,
        'apikey': API_KEY,
        'outputsize': output_size,
        'adjusted': adjusted,
        'extended_hours': extended_hours,
        'month': month
    }

    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"HTTP error: {response.status_code}")
            if retries < max_retries:
                time.sleep(60)
                return get_time_series_data_for_month(function, symbol, interval, month, output_size, adjusted, extended_hours, retries + 1)
            else:
                print("Max retries reached. Exiting.")
                return None

        data = response.json()

        # Check for API rate limit or error messages
        if "Note" in data:
            print("Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(60)
            if retries < max_retries:
                return get_time_series_data_for_month(function, symbol, interval, month, output_size, adjusted, extended_hours, retries + 1)
            else:
                print("Max retries reached. Exiting.")
                return None
        elif "Error Message" in data:
            print(f"Error fetching data for {month}: {data['Error Message']}")
            return None

        # Extract time series data
        time_series_key = next((key for key in data.keys() if 'Time Series' in key), None)
        if time_series_key:
            time_series = data[time_series_key]
            # Convert time series to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = [col.split('. ')[1] for col in df.columns]  # Clean up column names
            df = df.apply(pd.to_numeric)  # Convert all columns to numeric values
            return df

    except Exception as e:
        print(f"An error occurred: {e}")
        if retries < max_retries:
            time.sleep(5)
            return get_time_series_data_for_month(function, symbol, interval, month, output_size, adjusted, extended_hours, retries + 1)
        else:
            print("Max retries reached. Exiting.")
            return None


def get_time_series_data_by_date_range(function, symbol, interval, start_date, end_date, output_size='full', adjusted='true', extended_hours='true'):
    """Fetches time series data for a specified date range and stitches the monthly data together.
    
    Args:
        function (str): Alpha Vantage function (e.g., 'TIME_SERIES_INTRADAY').
        symbol (str): The stock ticker (e.g., 'IBM').
        interval (str): Data interval (e.g., '5min', '15min'). Required for intraday functions.
        start_date (datetime): Start date as a datetime object.
        end_date (datetime): End date as a datetime object.
        output_size (str): Output size ('compact' or 'full'). Defaults to 'full'.
        adjusted (str): 'true' or 'false' for adjusted data. Defaults to 'true'.
        extended_hours (str): 'true' or 'false' to include extended hours. Defaults to 'true'.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data for the specified date range.
    """
    # Ensure that start_date and end_date are datetime objects
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("start_date and end_date must be datetime objects")

    all_data = []

    current_date = start_date
    while current_date <= end_date:
        # Get the month in 'YYYY-MM' format
        month = current_date.strftime('%Y-%m')
        print(f"Fetching data for {month}...")
        
        # Fetch data for the current month
        df_month = get_time_series_data_for_month(function, symbol, interval, month, output_size, adjusted, extended_hours)
        
        if df_month is not None:
            all_data.append(df_month)
        else:
            print(f"No data retrieved for {month}")

        # Move to the next month
        current_date += relativedelta(months=1)

    # Concatenate all monthly data
    if all_data:
        df_combined = pd.concat(all_data, axis=0)
        df_combined.sort_index(inplace=True)
        return df_combined
    else:
        print("No data was retrieved for the specified date range.")
        return None


# Example usage
if __name__ == "__main__":
    function = 'TIME_SERIES_INTRADAY'
    symbol = 'AAPL'
    interval = '60min'
    start_date = datetime(2009, 1, 1)
    end_date = datetime(2012, 3, 31)
    output_size = 'full'
    adjusted = 'true'
    extended_hours = 'true'

    df = get_time_series_data_by_date_range(function, symbol, interval, start_date, end_date, output_size, adjusted, extended_hours)

    if df is not None:
        print("Data fetched successfully:")
        print(df.head())
    else:
        print("Failed to retrieve data.")
