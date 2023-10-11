'''
Implementing a moving-average strategy on the BITCOIN price.
Goal is to understand: when the day starts, should I sell or BUY?
'''
import torch
import numpy as np
from Historic_Crypto import HistoricalData
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt


# Import the BITCOIN data corresponding to the past 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

daily = 60*60*24
currency = "BTC-USD"

candles = HistoricalData(
    ticker=currency,
    granularity=daily,
    start_date=start_date.strftime("%Y-%m-%d-%H-%m"),
    end_date=end_date.strftime("%Y-%m-%d-%H-%M"),
    verbose=False
    ).retrieve_data()


def get_moving_average(df, window):
    timespan = df.shape[0] - window + 1
#    tseries = torch.zeros(timespan)
    # The moving-average vector is initially filled with -1
#    tseries = torch.ones(df.shape[0]) * -1
    tseries = torch.zeros(df.shape[0])
    for nth in range(timespan):
        # and then right values are computed from a suitable starting point on
        tseries[nth + window - 1] = df[nth : nth+window].mean()
    ready = tseries.detach().numpy()
    ready[:window] = np.NaN
    return ready
#---

slow_window = 15
fast_window = 5

# Compute fast moving-averages of the CLOSED prices
ma_f = get_moving_average(candles["close"], fast_window)

# Compute slow moving averages of the CLOSED prices
ma_s = get_moving_average(candles["close"], slow_window)

# Add to the dataset the two new time series
candles["ma_f"] = ma_f
candles["ma_s"] = ma_s

# Plot?
ma_f_plot = mpf.make_addplot(candles["ma_f"],
                linestyle='dotted', alpha=0.5,color='orange',secondary_y=False)
ma_s_plot = mpf.make_addplot(candles["ma_s"],
                linestyle='dotted', alpha=0.5, color='blue',secondary_y=False)
mpf.plot(candles, addplot=[ma_f_plot, ma_s_plot], type="candle", style="yahoo")


# Check when they cross and elaborate a strategy
# First, cut the dataset to the first common point, should be after
# slow_window days
print("Proceeding with trading strategy...")
wallet = 0.
candles = candles[15:]
if candles["ma_f"][0] > candles["ma_s"][0]:
    f_higher_s = True
else:
    f_higher_s = False

for nth in range(1,candles.shape[0]):
    buy_price = candles["open"][nth]
    sell_price = candles["close"][nth]
    delta = 0.

    if f_higher_s:
        if candles["ma_f"][nth] < candles["ma_s"][nth]:
            # Turning point!
            print(f"Turning point at day {nth}: f becomes lower than s")
            f_higher_s = False
        else:
            print("Prices are decreasing: sell, then buy.")
            delta = buy_price - sell_price


    if not f_higher_s:
        if candles["ma_f"][nth] > candles["ma_s"][nth]:
            print(f"Turning point at day {nth}: f becomes higher than s")
            f_higher_s = True
        else:
            print("Prices are increasing: buy, then sell.")
            delta = -buy_price + sell_price

    wallet += delta
    print(f"Day{nth}, wallet {wallet:.2f}")

print(f"Final wallet value: {wallet:.2f}")
