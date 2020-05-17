import pandas as pd
from data_combiner_month import get_this_month

def add_lookback_prefix(data,suffix):
    data = data.add_suffix(suffix)
    data.rename(columns={'Date'+suffix:'Date'},inplace=True)
    return data


def get_day_attributes(add_look_back = 5):

    sp500 = 'SP500.csv'
    sp500_df = pd.read_csv(sp500)
    sp500_df.drop(columns=['Open', 'Adj Close'], inplace=True)  # SP500 : Date | High | Low | Close | Volume

    vix = 'VIX.csv'
    vix_df = pd.read_csv(vix)
    vix_df.drop(columns=['Open', 'Adj Close', 'Volume'], inplace=True)
    vix_df.rename(columns={'Close': 'Vix_Close','High':'Vix_High', 'Low':'Vix_Low'}, inplace=True) #VIX : Date | Vix_Close| Vix_High | Vix_Low

    apa = 'APA.csv'
    apa_df = pd.read_csv(apa)
    apa_df.drop(columns=['Open', 'Adj Close'],inplace=True)
    apa_df.rename(columns={'Close': 'Apa_Close','High':'Apa_High', 'Low':'Apa_Low','Volume':'Apa_Volume'}, inplace=True)

    gld = 'GL.csv'
    gld_df = pd.read_csv(gld)
    gld_df.drop(columns=['Open', 'Adj Close'], inplace=True)
    gld_df.rename(columns={'Close': 'Gld_Close', 'High': 'Gld_High', 'Low': 'Gld_Low', 'Volume': 'Gld_Volume'},inplace=True)

    nasdq = 'nasdq.csv'
    nasdq_df = pd.read_csv(nasdq)
    nasdq_df.drop(columns=['Open', 'Adj Close'], inplace=True)
    nasdq_df.rename(columns={'Close': 'Nas_Close', 'High': 'Nas_High', 'Low': 'Nas_Low', 'Volume': 'Nas_Volume'},inplace=True)

    tnx = 'Treasury_Yield_10_Years.csv'
    tnx_df = pd.read_csv(tnx)
    tnx_df.drop(columns=['Open', 'Adj Close','Volume'], inplace=True)
    tnx_df.rename(columns={'Close': 'Tnx_Close', 'High': 'Tnx_High', 'Low': 'Tnx_Low'},inplace=True)
    tnx_df.dropna(inplace=True)

    #Attributes inner join
    day_combiner = pd.merge(sp500_df, vix_df).merge(apa_df).merge(gld_df).merge(nasdq_df).merge(tnx_df)
    if add_look_back == 0:
        day_combiner['DateMonthFormat'] = day_combiner['Date'].apply(lambda x: get_this_month(x))
        return day_combiner

    #add lookback for 5 days
    day_combiner_one = day_combiner.copy()
    day_combiner_one['Date'] = day_combiner_one['Date'].shift(periods=-1)
    day_combiner_one.dropna(inplace=True)
    day_combiner_one = add_lookback_prefix(day_combiner_one,'_1')

    day_combiner_two = day_combiner.copy()
    day_combiner_two['Date'] = day_combiner_two['Date'].shift(periods=-2)
    day_combiner_two.dropna(inplace=True)
    day_combiner_two = add_lookback_prefix(day_combiner_two, '_2')

    if add_look_back == 2:
        day_combiner_tag = day_combiner[['Date', 'Close']].copy().rename(columns={'Close': 'Price_Day'})
        day_combiner = pd.merge(day_combiner_one, day_combiner_two).merge(
            day_combiner_tag)  # day combiner of the same day is for debugging

        # support monthly format for days/months join
        day_combiner['DateMonthFormat'] = day_combiner['Date'].apply(lambda x: get_this_month(x))
        return day_combiner
    day_combiner_three = day_combiner.copy()
    day_combiner_three['Date'] = day_combiner_three['Date'].shift(periods=-3)
    day_combiner_three.dropna(inplace=True)
    day_combiner_three = add_lookback_prefix(day_combiner_three, '_3')

    day_combiner_four = day_combiner.copy()
    day_combiner_four['Date'] = day_combiner_four['Date'].shift(periods=-4)
    day_combiner_four.dropna(inplace=True)
    day_combiner_four = add_lookback_prefix(day_combiner_four, '_4')

    day_combiner_five = day_combiner.copy()
    day_combiner_five['Date'] = day_combiner_five['Date'].shift(periods=-5)
    day_combiner_five.dropna(inplace=True)
    day_combiner_five = add_lookback_prefix(day_combiner_five, '_5')

    day_combiner_tag= day_combiner[['Date', 'Close']].copy().rename(columns={'Close': 'Price_Day'})
    day_combiner = pd.merge(day_combiner_one,day_combiner_two).merge(day_combiner_three).merge(day_combiner_four).merge(day_combiner_five).merge(day_combiner_tag) #day combiner of the same day is for debugging

    # support monthly format for days/months join
    day_combiner['DateMonthFormat'] = day_combiner['Date'].apply(lambda x: get_this_month(x))


    return day_combiner