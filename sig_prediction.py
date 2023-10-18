'''
Try to PREDICT values using a Windowed-Signature approach.
'''
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from libasc import to_log_returns, augment_db, single_ts_to_windowed
from libasc import relerr
from libqoi import mix_and_split
import signatory as sg


n_times = 2000
window = 20
sig_depth = 5


# (1) CONSTRUCTING THE 1-DIM TIME SERIES
#x_tmp = torch.ones(n_times,)
x_tmp = torch.abs(torch.normal(0., 1., (n_times,)))


#
# (2) BUILDING THE DATASET FROM THE TIME SERIES
#

# decomposing the dataset into multiple "window chunks"
db0 = single_ts_to_windowed(x_tmp, window)

# removing the last "window chunk", since we do not know its label.
# (label = next price, and we don't know the future)
n_samples = db0.shape[0] - 1
db0_without_last = db0[:n_samples]
assert (db0_without_last.shape[0] == (n_times - window))
# as already said, labels (y) are the next price after the window chunk
y_data = torch.zeros(n_times - window)
for nth in range(n_times - window):
    y_data[nth] = x_tmp[nth + window]
print(f"Windowed dataset: {db0_without_last}, {y_data}")
y_data = y_data.reshape(-1, 1)
# Convert data into log returns
db1 = to_log_returns(db0_without_last)
# Augment the dataset
db2 = augment_db(db1)
# Take its signature, constructing then the data on which to train
x_data  = sg.signature(db2, depth = sig_depth)
# At this point, we have a ready dataset of x_data and y_data
# Shuffle everything and be ready for training!
x_train, y_train, x_val, y_val = mix_and_split(x_data, y_data)


#
# (3) TRAIN A LINEAR OPTIMIZER
#
sig_len = x_train[0].shape[0]
model = nn.Linear(sig_len, 1)

# Store the overall predicted results
y_initial_pred = model(x_data).detach()
# Error estimation
start_mre = relerr(y_data, y_initial_pred, 1e-2)
print(f"Starting mean Relative Error: {start_mre:.1}%")
# Set Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()
n_epochs = 30000
# History of loss function values
hist_loss_train = torch.ones(n_epochs)
hist_loss_val = torch.ones(n_epochs)

# Training Loop
for nth in range(n_epochs):
    # Perform optimization
    optimizer.zero_grad()
    y_train_pred = model(x_train)
    loss_train = loss_fn(y_train_pred, y_train)

    with torch.no_grad():
        y_val_pred = model(x_val)
        loss_val = loss_fn(y_val_pred, y_val)
        if n_epochs % 1000 == 0:
            print(f"{nth+1}/{n_epochs} t: {loss_train.item():.3e} ", end=' ')
            print(f"v:{loss_val.item():.3e}")
        hist_loss_train[nth] = loss_train.item()
        hist_loss_val[nth] = loss_val.item()

    loss_train.backward()
    optimizer.step()
#---

print(f"Training ended!")
plt.plot(hist_loss_train[100:], label='train')
plt.plot(hist_loss_val[100:], label='val')
plt.grid()
plt.legend()
plt.title("Loss function evolution")
plt.show()

#
# (4) EVALUATE THE RESULTS BY COMPUTING THE MEAN RELATIVE AND SQUARE ERROR
#
# Store the overall predicted results
y_pred = model(x_data).detach()

# Error estimation
mre = relerr(y_data, y_pred, 1e-2)
print(f"Starting mean Relative Error: {start_mre:.1f}%")
print(f"Final Mean Relative Error: {mre:.1f}%")
mse = loss_fn(y_data, y_pred)
print(f"Mean Square Error: {mse:.3e}")

# max and min used to scale the plot accordingly
y_min = torch.min(y_data)
y_max = torch.max(y_data)
plt.scatter(y_data[:,0], y_pred[:,0], label='true data VS predicted',
                color='orange')
plt.plot(torch.linspace(y_min, y_max, 10),
            torch.linspace(y_min, y_max, 10),
            label='diagonal target',
            linestyle='dashed', color='blue')
plt.title(f"Attempt of Sig-Prediction [mre = {mre:.1f}%]")
plt.legend()
plt.grid()
plt.show()


#
# (5) VERIFY THE TRADING STRATEGY VIA BACKTRACKING
#

# "ratio" represents how many stock are going to buy and sell each time
ratio = 1.0
transaction_cost = 0.1

wallet = 0.
for nth in range(y_data.shape[0]):
    # Determine the current value, the true next one and the predicted
    x_next = y_data[nth].item()
    x_pred = y_pred[nth].item()
    t = nth + window - 1
    x_now = x_tmp[t]
    print(f"time {t}. Curr: {x_now:.2f} Pred: {x_pred:.2f} True: {x_next:.2f}")
    # Determine if buy, sell or stay idle
    percentage = x_pred / 100. * mre
    up_val = x_pred + percentage
    low_val = x_pred - percentage
    profit = 0
    status = "TOO UNCERTAIN"
    mycolor="grey"
    action="IDLE"
    # If the lowest predicted value is higher than the current,
    # even considering the transaction costs...
    if low_val - 2*transaction_cost - x_now > 0:
        #...buy a stock percentage, sell the instant later
        status = "INCREASES"
        mycolor="green"
        action="BUY"
        profit = x_next - x_now - 2*transaction_cost
    # If the highest predicted value is lower than the current,
    # even considering the transaction costs...
    elif x_now - up_val - 2*transaction_cost > 0:
        # ...sell a stock percentage
        status = "DECREASES"
        mycolor="red"
        action="SELL"
        profit = x_now - 2*transaction_cost - x_next
    wallet += profit
    print(f" --- action: {action}. Curr wallet: {wallet:.3f}")
   

print(f"Final wallet: {wallet:.3f}")
print(f"(--- (considering transaction costs of {transaction_cost:.3f} ---)")
#input("OK?")

#
# DETERMINE WHICH NEXT ACTION TO TAKE
#

tmp_to_predict = augment_db(to_log_returns(db0[-1].reshape(1, -1)))
next_to_predict = sg.signature(tmp_to_predict, depth = sig_depth)
predicted_val = model(next_to_predict).detach().item()
percentage = predicted_val / 100. * mre
up_val = predicted_val + percentage
low_val = predicted_val - percentage
print(f"PREDICTED NEXT VALUE: {predicted_val:.2f} [{low_val}, {up_val}]")

status = "TOO UNCERTAIN"
mycolor="grey"
action="IDLE"
expected_profit = 0.
if low_val - 2*transaction_cost > x_tmp[-1]:
    status = "INCREASES"
    mycolor="green"
    action="BUY"
    expected_profit = low_val - 2*transaction_cost - x_tmp[-1]
elif up_val + 2*transaction_cost < x_tmp[-1]:
    status = "DECREASES"
    mycolor="red"
    action="SELL"
    expected_profit = - up_val - 2*transaction_cost + x_tmp[-1]
print(f"{action}: price {status}: {x_tmp[-1]:.2f} -> {predicted_val:.2}")
print(f"Expected profit: {expected_profit:.3f}")

x_mean = x_tmp.mean()
time_to_show = 10
plt.plot(range(time_to_show), x_tmp[-time_to_show:], color='orange')
plt.plot(time_to_show, predicted_val, color=mycolor, marker="X")
plt.plot(time_to_show, up_val, color=mycolor, marker="v")
plt.plot(time_to_show, low_val, color=mycolor, marker="^")
plt.axhline(y = x_mean, color='blue', linestyle='dashed')
plt.title("Price prediction using Sig-Windowing")
plt.grid()
plt.show()
