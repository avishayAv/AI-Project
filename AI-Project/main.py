
import pandas

def get_dates_of_crises(increase_sequence, recovery, crisis_percentage):
    sp_data = pandas.read_csv("SP500Daily_long.csv")[['Date', 'Open']].to_numpy() # Date | Open (price)
    start_day = 0
    crisis = []
    while start_day < len(sp_data):
        start_day_date = sp_data[start_day][0]
        start_day_price = sp_data[start_day][1]
        min_day_price = sp_data[start_day][1]
        increase_max_sequence = 1
        # Calc losing sequence
        while (start_day + increase_max_sequence < len(sp_data) and
               (sp_data[start_day+increase_max_sequence][1] <= sp_data[start_day+increase_max_sequence-1][1] or
               sp_data[start_day+increase_max_sequence][1] <= sp_data[start_day][1] * (1-recovery))):
            min_day_price = min(min_day_price, sp_data[start_day + increase_max_sequence][1])
            increase_max_sequence = increase_max_sequence + 1
        # make sure losing sequence is long enough
        if increase_max_sequence >= increase_sequence and start_day + increase_max_sequence < len(sp_data) and min_day_price * (1+crisis_percentage) <= start_day_price:
            crisis.append((sp_data[start_day][0], sp_data[start_day+increase_max_sequence-1][0]))
        start_day = start_day + increase_max_sequence
    print (len(crisis))
    print (crisis)

crisis_percentage = {'small': 5, 'medium': 10, 'big': 15}
crisis_increase_sequence = {'short': 5, 'medium': 10, 'long': 15}
crisis_recovery = {'partially': 0.05, 'full': 0}
result = get_dates_of_crises(5, 0.045, 0.025)