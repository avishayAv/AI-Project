import pandas as pd
from datetime import datetime
from datetime import timedelta

# Given a date string, create a date object and calculate previous month.
# used for data that released in the first day of the month - which will be applied on the previous month.
def get_previous_month(current_date):
    curr = datetime.strptime(current_date, '%Y-%m-%d')
    delta = timedelta(weeks=4)
    return (curr-delta).year, (curr-delta).month

# Given a date string, create a date object and return it.
def get_this_month(current_date):
    curr = datetime.strptime(current_date, '%Y-%m-%d')
    return curr.year, curr.month

# Given a date (year, month), calc the next month in the same format.
def get_next_month(current_date):
    year, month = current_date
    next_month = (month + 1) % 12
    if next_month == 0:
        next_month = 12
    next_year = year if month < 12 else year+1
    return next_year, next_month

def get_month_attributes():
    shiller_pe_ratio = 'MULTPL-SHILLER_PE_RATIO_MONTH.csv'
    dividend = 'MULTPL-SP500_DIV_MONTH.csv'
    dividend_yield = 'MULTPL-SP500_DIV_YIELD_MONTH.csv'
    earnings_yield = 'MULTPL-SP500_EARNINGS_YIELD_MONTH.csv'
    infladj = 'MULTPL-SP500_INFLADJ_MONTH.csv'
    pe_ratio = 'MULTPL-SP500_PE_RATIO_MONTH.csv'

    # Read csv, get valid monthly date format and refactor names before joining.
    shiller_pe_ratio_df = pd.read_csv(shiller_pe_ratio)
    shiller_pe_ratio_df['Date'] = shiller_pe_ratio_df['Date'].apply(lambda x : get_previous_month(x))
    shiller_pe_ratio_df.rename(columns={'Value':'ShillerPEValue'}, inplace=True)

    dividend_df = pd.read_csv(dividend)
    dividend_df['Date'] = dividend_df['Date'].apply(lambda x : get_this_month(x))
    dividend_df.rename(columns={'Value':'DividendValue'}, inplace=True)

    dividend_yield_df = pd.read_csv(dividend_yield)
    dividend_yield_df['Date'] = dividend_yield_df['Date'].apply(lambda x : get_this_month(x))
    dividend_yield_df.rename(columns={'Value':'DividendYieldValue'}, inplace=True)

    earnings_yield_df = pd.read_csv(earnings_yield)
    earnings_yield_df['Date'] = earnings_yield_df['Date'].apply(lambda x : get_previous_month(x))
    earnings_yield_df.rename(columns={'Value':'EarningsYieldValue'}, inplace=True)

    infladj_df = pd.read_csv(infladj)
    infladj_df['Date'] = infladj_df['Date'].apply(lambda x : get_previous_month(x))
    infladj_df.rename(columns={'Value':'InflationValue'}, inplace=True)

    pe_ratio_df = pd.read_csv(pe_ratio)
    pe_ratio_df['Date'] = pe_ratio_df['Date'].apply(lambda x : get_previous_month(x))
    pe_ratio_df.rename(columns={'Value':'PERatioValue'}, inplace=True)

    month_combiner = pd.merge(shiller_pe_ratio_df, dividend_df).merge(dividend_yield_df).merge(earnings_yield_df).merge(infladj_df).merge(pe_ratio_df)
    month_combiner.rename(columns={'Date':'DateMonthFormat'}, inplace=True)

    # support previous month data for each month.
    prev_month_combiner = month_combiner.copy()
    prev_month_combiner.rename(columns={'ShillerPEValue': 'PrevMonthShillerPEValue',
                                'DividendValue': 'PrevMonthDividendValue',
                                'DividendYieldValue': 'PrevMonthDividendYieldValue',
                                'EarningsYieldValue': 'PrevMonthEarningsYieldValue',
                                'InflationValue': 'PrevMonthInflationValue',
                                'PERatioValue': 'PrevMonthPERatioValue'}, inplace=True)
    prev_month_combiner['DateMonthFormat'] = prev_month_combiner['DateMonthFormat'].apply(lambda x : get_next_month(x))

    # support prevprev month data for each month.
    prev_prev_month_combiner = month_combiner.copy()
    prev_prev_month_combiner.rename(columns={'ShillerPEValue': 'PrevPrevMonthShillerPEValue',
                                        'DividendValue': 'PrevPrevMonthDividendValue',
                                        'DividendYieldValue': 'PrevPrevMonthDividendYieldValue',
                                        'EarningsYieldValue': 'PrevPrevMonthEarningsYieldValue',
                                        'InflationValue': 'PrevPrevMonthInflationValue',
                                        'PERatioValue': 'PrevPrevMonthPERatioValue'}, inplace=True)
    prev_prev_month_combiner['DateMonthFormat'] = prev_prev_month_combiner['DateMonthFormat'].apply(lambda x: get_next_month(get_next_month(x)))

    month_combiner_with_lookback = pd.merge(month_combiner, prev_month_combiner).merge(prev_prev_month_combiner).drop(columns={'ShillerPEValue','DividendValue','DividendYieldValue','EarningsYieldValue','InflationValue','PERatioValue'})


    return month_combiner_with_lookback

