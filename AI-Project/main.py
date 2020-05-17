from data_extractor import get_dates_of_crises
from data_combiner_month import *
from data_combiner_day import *
import pandas as pd
from feature_selection import *
import os
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


def main():
    crisis_percentage = {'small': 5, 'medium': 10, 'big': 15}
    crisis_increase_sequence = {'short': 5, 'medium': 10, 'long': 15}
    crisis_recovery = {'partially': 0.05, 'full': 0}
    crisis = get_dates_of_crises(5, 0.04, 0.025, 1) #best_option
    #print(len(crisis))
    #print(crisis)

    #Attributes INNER JOIN
    combiner = pd.merge(get_day_attributes(), get_month_attributes())
    combiner = pd.merge(combiner,crisis_to_df(crisis))
    # add label - min max normlization
    combiner['label']=combiner.apply(lambda row: (row['Price_Max'] -row['Price_Day'])/(row['Price_Max'] -row['Price_Min']) if (row['Price_Max'] - row['Price_Min']) else 0 , axis=1)
    combiner.drop(columns=['Price_Max','Price_Day','Price_Min','DateMonthFormat'],inplace=True)
    combiner.to_csv(os.getcwd()+'/combiner.csv')
#-------------------------------------------------------------------------------
    X = combiner.loc[:, combiner.columns != 'label'].drop(columns=['Date'])
    y = (combiner['label'] * 10).astype(int)


    # normalization
    normalized_X = scale_min_max(X)
    standardized_X = normalization(X)


    #find k best with orignial data ,normalized and standardized
    """print("KBest,X")
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
    findKExtraTree_feature_selection(standardized_X, y, standardized_X.shape[1])"""

if __name__ == "__main__":
    main()