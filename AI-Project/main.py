from data_extractor import get_dates_of_crises
from data_combiner_month import *
from data_combiner_day import *
import pandas as pd

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

    #debug
    #for i in combiner.columns:
    #   print(i)




if __name__ == "__main__":
    main()