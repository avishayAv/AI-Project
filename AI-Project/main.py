import pandas

# Features :
# increase_sequence (int) - minimum market days of "losing sequence" to tag as crisis.
# recovery (%) - when the markets are back to price(start_day)*(1-recovery), crisis is over.
# crisis_percentage (%) - minimum damage to tag as crisis.
# allow_day_fix (0/1) - ignore a "fix day" in calculation of losing sequence.
def get_dates_of_crises(increase_sequence, recovery, crisis_percentage, allow_day_fix):
    sp_data = pandas.read_csv("SP500Daily_long.csv")[['Date', 'Open']].to_numpy()  # Date | Open (price)
    start_day = 0
    crisis = []
    while start_day < len(sp_data):
        #test potential starting day for crisis
        start_day_date = sp_data[start_day][0]
        start_day_price = sp_data[start_day][1]
        min_day_price = sp_data[start_day][1]
        increase_max_sequence = 1
        count_day_fix = 0
        # Calc losing sequence. triggers (one is enough):
        # 1. chart is decreasing.
        # 2. chart is increasing but recovery still not completed. means : price(current) <= price(start_day)*(1-recovery)
        # 3. chart is increasing but there's a one day fix (common in crises).
        while (start_day + increase_max_sequence < len(sp_data) and
               (sp_data[start_day + increase_max_sequence][1] <= sp_data[start_day + increase_max_sequence - 1][1] or
                sp_data[start_day + increase_max_sequence][1] <= sp_data[start_day][1] * (1 - recovery) or
                (allow_day_fix and count_day_fix < 1))):
            # if trigger is #3, use it once.
            if (sp_data[start_day + increase_max_sequence][1] > sp_data[start_day + increase_max_sequence - 1][1] and
                    sp_data[start_day + increase_max_sequence][1] > sp_data[start_day][1] * (1 - recovery)):
                count_day_fix = count_day_fix + 1
            min_day_price = min(min_day_price, sp_data[start_day + increase_max_sequence][1])
            increase_max_sequence = increase_max_sequence + 1
        # make sure losing sequence is long enough and crisis is big enough.
        if increase_max_sequence >= increase_sequence and \
                start_day + increase_max_sequence < len(sp_data) and \
                min_day_price * (1 + crisis_percentage) <= start_day_price:
            crisis.append(
                (sp_data[start_day][0], sp_data[start_day + increase_max_sequence - 1][0], increase_max_sequence))
        start_day = start_day + increase_max_sequence
    print(len(crisis))
    print(crisis)


crisis_percentage = {'small': 5, 'medium': 10, 'big': 15}
crisis_increase_sequence = {'short': 5, 'medium': 10, 'long': 15}
crisis_recovery = {'partially': 0.05, 'full': 0}
result = get_dates_of_crises(5, 0.04, 0.025, 1)
