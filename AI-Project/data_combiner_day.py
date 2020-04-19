import pandas as pd
from data_combiner_month import get_this_month

def get_day_attributes():
    vix = 'VIX.csv'
    vix_df = pd.read_csv(vix)
    vix_df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    vix_df.rename(columns={'Close': 'Vix_Close'}, inplace=True) #VIX : Date | Vix_Close

    sp500 = 'SP500.csv'
    sp500_df = pd.read_csv(sp500)
    sp500_df.drop(columns=['Open', 'Adj Close'], inplace=True) #SP500 : Date | High | Low | Close | Volume

    #Attributes inner join
    day_combiner = pd.merge(sp500_df, vix_df)

    #support monthly format for days/months join
    day_combiner['DateMonthFormat'] = day_combiner['Date'].apply(lambda x: get_this_month(x))

    return day_combiner