from data_extractor import get_dates_of_crises
from data_combiner_month import *
from data_combiner_day import *
import pandas as pd

def main():
    crisis_percentage = {'small': 5, 'medium': 10, 'big': 15}
    crisis_increase_sequence = {'short': 5, 'medium': 10, 'long': 15}
    crisis_recovery = {'partially': 0.05, 'full': 0}
    crisis = get_dates_of_crises(5, 0.04, 0.025, 1) #best_option
    #print(len(crisis))
    #print(crisis)

    #Attributes INNER JOIN
    combiner = pd.merge(get_day_attributes(), get_month_attributes())
    # TODO : drop 'DateMonthFormat' attribute which used for merge only

    print (combiner)


if __name__ == "__main__":
    main()