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
window = 10
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
# (label = immediate next price, and we don't know the future)
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


optimizer = optim.Adam(model.parameters(), lr=1e-5)

# And the loss functions, canonical
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
# (4) EVALUATE THE RESULTS
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
if low_val > x_tmp[-1]:
    status = "INCREASES"
    mycolor="green"
    action="BUY"
elif up_val < x_tmp[-1]:
    status = "DECREASES"
    mycolor="red"
    action="SELL"
print(f"{action}: price {status}: {x_tmp[-1]:.2f} -> {predicted_val:.2}")

x_mean = x_tmp.mean()
time_to_show = 50
plt.plot(range(time_to_show), x_tmp[-time_to_show:], color='orange')
plt.plot(time_to_show, predicted_val, color=mycolor, marker="X")
plt.plot(time_to_show, up_val, color=mycolor, marker="v")
plt.plot(time_to_show, low_val, color=mycolor, marker="^")
plt.axhline(y = x_mean, color='blue', linestyle='dashed')
plt.title("Price prediction using Sig-Windowing")
plt.grid()
plt.show()
