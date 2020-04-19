import pandas

# remove manually bubbles affection from crises dates.
# input : start_day, end_day (which are about to be added to crises)
# output : bubble_end_day if a bubble is in the middle of the so called crises, -1 otherwise.
def handle_economic_bubble(start_day, end_day):
    handle_bubbles = [682 # 1930-09-22
                      ]
    for bubble in handle_bubbles:
        if bubble > start_day and bubble < end_day:
            return bubble
    return -1

# bear market : a situation that a market decrease by >20% and stays in the new level for a while.
# in this situation, we would like to make the recovery standards easier, since this is not realistic to expect the
# market to fully recover.
# unstable market : a situation that there's a lot of ups and downs that does not necessarily indicates the market's position.
# in this situation, we would like to make the recovery standards harder, since we don't want to tag a crisis for each and
# every ups and downs.
# output : 1 if not in a bear position and not an unstable market.
#          bear market - another coefficient (>1, depends on how deep the market has been hearted)
#          unstable market - another coefficient (<1, depends on how unstable it is)
def handle_crisis_before_bear_market_or_unstable_market(trading_day):
    # tuple structure is (start_period, end_period, coefficient)
    bear_markets = [(2623, 2666, 7)] # Jul 1938 - Sep 1938 - Decreased 25% (Comparing to 1937)
    unstable_markets = [(7419, 7628, 0.5)] #Sep 1957 - Jul 1958
    markets = bear_markets + unstable_markets
    for start_period, end_period, coefficient in markets:
        if trading_day > start_period and trading_day < end_period:
            return coefficient
    return 1

# Features :
# decrease_sequence_lmt (int) - minimum market days of "losing sequence" to tag as crisis.
# recovery (%) - when the markets are back to price(start_day)*(1-recovery), crisis is over.
# crisis_percentage (%) - minimum damage to tag as crisis.
# allow_day_fix (0/1) - ignore a "fix day" in calculation of losing sequence.
def get_dates_of_crises(decrease_sequence_lmt, recovery, crisis_percentage, allow_day_fix):
    sp_data = pandas.read_csv("SP500.csv")[['Date', 'Close']].to_numpy()  # Date | Open (price)
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
                sp_data[start_day + increase_max_sequence][1] <= sp_data[start_day][1] * (1 - (recovery * handle_crisis_before_bear_market_or_unstable_market(start_day + increase_max_sequence))) or
                (allow_day_fix and count_day_fix < 1))):
            # if trigger is #3, use it once.
            if (sp_data[start_day + increase_max_sequence][1] > sp_data[start_day + increase_max_sequence - 1][1] and
                    sp_data[start_day + increase_max_sequence][1] >
                    sp_data[start_day][1] * (1 - (recovery * handle_crisis_before_bear_market_or_unstable_market(start_day + increase_max_sequence)))):
                count_day_fix = count_day_fix + 1
            min_day_price = min(min_day_price, sp_data[start_day + increase_max_sequence][1])
            increase_max_sequence = increase_max_sequence + 1
        # make sure losing sequence is long enough and crisis is big enough.
        if increase_max_sequence >= decrease_sequence_lmt and \
                start_day + increase_max_sequence < len(sp_data) and \
                min_day_price * (1 + crisis_percentage) <= start_day_price:
            end_day = start_day + increase_max_sequence - 1
            bubble_inside_crisis = handle_economic_bubble(start_day, end_day)
            if bubble_inside_crisis == -1:
                crisis.append(
                    (sp_data[start_day][0], sp_data[start_day + increase_max_sequence - 1][0], increase_max_sequence))
            else:
                start_day = bubble_inside_crisis
                continue
        start_day = start_day + increase_max_sequence
    return crisis