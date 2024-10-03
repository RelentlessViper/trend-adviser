import yfinance as yf
from datetime import datetime
import pandas as pd

# Set up a ticker
SP_500 = yf.Ticker(ticker='SPY')

def __shift_by_days__(df: pd.DataFrame, shift_days_amount: int = 4) -> pd.DataFrame:
    handler_df = df.copy()
    
    handler_df['Close_-1'] = handler_df.loc[:, 'Close'].shift(1)
    for day_idx in range(2, shift_days_amount + 1):
        handler_df[f'Close_-{day_idx}'] = handler_df.loc[:, f'Close_-{day_idx - 1}'].shift(1)
    
    return handler_df.dropna()

def create_dataset(start_date: datetime, end_date: datetime, n_days: int = 4) -> pd.DataFrame:
    """
    Retrieve S&P 500 data and create dataset that contains the "Closing" price for each day in a following way:
    Close    Close_-1    Close_-2    Close_-3 ... Close_-n
    
    Parameters
    ----------
    start_date: datetime
        Starting date. For example: `datetime(year=1980, month=1, day=1)`.
    end_date: datetime
        Ending date. For example: `datetime(year=2024, month=9, day=1)`.
    n_days: int
        The amount of days that will be used to predict the price for the next day.
    
    Returns
    ----------
    dataframe: pd.DataFrame
        The closing price of a day with closing prices of "n" previous days.
    """
    
    data_params = {
        'start': start_date,
        'end': end_date,
        'period': '1d'
    }
    
    # Get historical market data
    data = SP_500.history(**data_params)
    data = data.drop(labels=['Stock Splits', 'Capital Gains', 'Dividends', 'Open', 'High', 'Low', 'Volume'], axis=1)
    
    preprocessed_data = __shift_by_days__(df=data, shift_days_amount=n_days)
    #print(preprocessed_data)
    return preprocessed_data