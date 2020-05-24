from data_extractor import get_dates_of_crises
from data_combiner_month import *
from data_combiner_day import *
import pandas as pd
from feature_selection import *
import os
from lstm4 import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from test import *
import collections,functools,operator
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go



def scale_min_max(df: pd.DataFrame) -> pd.DataFrame:
    min = df.min(axis=0)
    max = df.max(axis=0)
    df = (df- min) / (max - min)
    return df


def normalization( df: pd.DataFrame):
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    df = (df - mean) / std
    return df


#receive str date and return str next day
def get_next_day(current_date):
    curr = datetime.strptime(current_date, '%Y-%m-%d')
    delta = timedelta(days=1)
    curr=curr+delta
    return str(curr.date())

#create dataframe with crisis dates and price_min and price max for each crisis
def crisis_to_df(crisis_list):
    df = pd.DataFrame(columns=['Date','Price_Min','Price_Max'])
    i=0
    for begin,end,_,min_value,max_value in crisis_list:
        current_date = begin
        while current_date != get_next_day(end):
            df.loc[i]=[current_date,min_value,max_value]
            i=i+1
            current_date = get_next_day(current_date)
    return df

def test_feature_selection(combiner):
    X = combiner.loc[:, combiner.columns != 'label'].drop(columns=['Date'])
    y = (combiner['label'] * 10).astype(int)

    # normalization
    normalized_X = scale_min_max(X)
    standardized_X = normalization(X)

    # find k best with orignial data ,normalized and standardized
    print("KBest,X")
    findKBest_features_selection(X,y,X.shape[1])
    print("KBest,normalized_X")
    findKBest_features_selection(normalized_X,y,normalized_X.shape[1])
    print("KBest,standardized_X")
    findKBest_features_selection(standardized_X,y,standardized_X.shape[1])

    # find k best with ExtraTree data ,normalized and standardized
    print("KExtra,X")
    findKExtraTree_feature_selection(X,y,X.shape[1])
    print("KExtra,normalized_X")
    findKExtraTree_feature_selection(normalized_X, y, normalized_X.shape[1])
    print("KExtra,standardized_X")
    findKExtraTree_feature_selection(standardized_X, y, standardized_X.shape[1])

def csv_create():
    crisis_percentage = {'small': 5, 'medium': 10, 'big': 15}
    crisis_increase_sequence = {'short': 5, 'medium': 10, 'long': 15}
    crisis_recovery = {'partially': 0.05, 'full': 0}
    crisis = get_dates_of_crises(5, 0.04, 0.025, 1)

    # full features periodically with label
    combiner = pd.merge(get_day_attributes(add_look_back=5), get_month_attributes(add_look_back=2))
    combiner = pd.merge(combiner, crisis_to_df(crisis))
    # add label - min max normlization
    combiner['label'] = combiner.apply(
        lambda row: (row['Price_Max'] - row['Price_Day']) / (row['Price_Max'] - row['Price_Min']) if (
                    row['Price_Max'] - row['Price_Min']) else 0, axis=1)
    combiner.drop(columns=['Price_Max', 'Price_Day', 'Price_Min', 'DateMonthFormat'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_full.csv')

    # full features periodically without label
    combiner = pd.merge(get_day_attributes(add_look_back=5), get_month_attributes(add_look_back=2))
    combiner = pd.merge(combiner, crisis_to_df(crisis))
    combiner.drop(columns=['Price_Max', 'Price_Day', 'Price_Min', 'DateMonthFormat'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_full_noLabel.csv')

    # continuous days without labels (without crisis filter)
    combiner = pd.merge(get_day_attributes(add_look_back=5), get_month_attributes(add_look_back=2))
    combiner.drop(columns=['Price_Day', 'DateMonthFormat'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_noLabel_noCrisis.csv')

    # continuous days without monthly features
    combiner = get_day_attributes(add_look_back=5)
    combiner.drop(columns=['DateMonthFormat', 'Price_Day'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_noMonth.csv')

    # continuous days without look back
    combiner = pd.merge(get_day_attributes(add_look_back=0), get_month_attributes(add_look_back=0))
    combiner.drop(columns=['DateMonthFormat'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_noLookback.csv')

    # continuous days without look back and without month
    combiner = get_day_attributes(add_look_back=0)
    combiner.drop(columns=['DateMonthFormat'], inplace=True)
    combiner.to_csv(os.getcwd() + '/combiner_noMonth_noLookback.csv')




def main():
    # create all csv
    csv_create()

    #part 1 try simple algorithms with feature selection
    #combiner = pd.read_csv('combiner_full.csv')
    #test_feature_selection(combiner)





if __name__ == "__main__":
    #main()

    #lstm_experiment('combiner_full_noLabel.csv',2000,epochs=30)
    #firts experiments

    #prediction1 = lstm_experiment('combiner_noMonth_noLookback.csv', 6000, epochs=20)
    #prediction2 = lstm_experiment('combiner_noLookback.csv', 6000, epochs=25)

    """data = pd.read_csv('combiner_noLookback.csv')

    date = data['Date'].values
    close = data['Close'].values

    date_train, date_test = date[:6000], date[6000:]
    close_train, close_test = close[:6000], close[6000:]

    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction1,
        mode='lines',
        name='Prediction_Without_Month'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=prediction2,
        mode='lines',
        name='Prediction_With_Month'
    )
    trace4 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    layout = go.Layout(
        title="S&P500",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    fig = go.Figure(data=[trace1, trace2,trace3, trace4], layout=layout)
    fig.show()"""

    data = pd.read_csv('combiner_noLookback.csv')

    date = data['Date'].values
    close = data['Close'].values

    date_train,date_test = date[:6000] ,date[6000:]
    close_train,close_test = close[:6000] ,close[6000:]

    epochs_list = list(range(18,32,4))
    batch_list = [32,48,64,72,96]
    units_list = list(range(30,90,10))
    for epoch in [26]:
        for batch in [32]:
            for unit in [70]:
                prediction = lstm_experiment('combiner_noLookback.csv', 6000, epochs=epoch,batch_size =batch,units=unit)
                name= f'epochs = {epoch} ,batch = {batch} , unit = {unit}'
                print(name)
                trace1 = go.Scatter(
                    x=date_train,
                    y=close_train,
                    mode='lines',
                    name='Data'
                )
                trace2 = go.Scatter(
                    x=date_test,
                    y=prediction,
                    mode='lines',
                    name='Prediction_With_Month'
                )
                trace3 = go.Scatter(
                    x=date_test,
                    y=close_test,
                    mode='lines',
                    name='Ground Truth'
                )
                layout = go.Layout(
                    title=name,
                    xaxis={'title': "Date"},
                    yaxis={'title': "Close"}
                )
                fig = go.Figure(data=[trace1,trace2,trace3], layout=layout)
                fig.show()