# Import necessary dependencies
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
import pickle
from dataset import create_dataset

app = FastAPI()

# Load pre-trained XGBRegressor
model_data = None
model = None # Main XGBRegressor model
n_days = None # The amount of days that will be used to predict the stock price
with open('models/xgbregressor_v1.pkl', 'rb') as file:
    model_data = pickle.load(file)

if model_data is not None:
    model = model_data['model']
    n_days = model_data['n_days']
else:
    raise Exception('"xgbregressor_v1.pkl" is not found.')

@app.post('/predict/')
async def predict(date: dict[str, str]): #-> pd.core.frame.DataFrame:
    """
    Generate predictions for a given time period.

    Parameters
    ----------
    date: dict[str, str]
        Dictionary that contains 2 variables:\n
            `start_date`: str
                Starting date in a format: `YYYY.MM.DD`.\n
            `end_date`: str
                Ending date in a format: `YYYY.MM.DD`.
        start_date must be `<` end_date.
    
    Returns
    ----------
    dataframe: pd.DataFrame
        Dataframe that contains actual "Closing" price and predicted "Closing" price.
    """

#date = {
#    'start_date': "2024.08.01",
#    'end_date': "2024.09.01"
#}

    start_date_idxs = list(map(int, date['start_date'].split('.')))
    start_date = datetime(year=start_date_idxs[0], month=start_date_idxs[1], day=start_date_idxs[2])

    end_date_idxs = list(map(int, date['end_date'].split('.')))
    end_date = datetime(year=end_date_idxs[0], month=end_date_idxs[1], day=end_date_idxs[2])

    # Retrieve main dataset
    main_dataset = create_dataset(start_date=start_date, end_date=end_date, n_days=n_days)
    target = main_dataset.loc[:, ['Close']]
    main_dataset = main_dataset.drop(labels=['Close'], axis=1)

    # Generate predictions
    predictions = model.predict(main_dataset)

    # Concatenate predictions and target
    dataset = target.copy()
    dataset['Predicted'] = predictions
    
    dataset = dataset.reset_index()  # Reset index to include date as a column
    dataset = dataset.to_dict(orient='records')
    
    return dataset