'''
In this script we approximate the MAX operator on a time series dataset
by following two approaches:
    - a simple linear model on the raw given data;
    - a simple linear model on the signature of data.
The second one should be theoretically more accurate. We check numerically.
'''
import torch
import signatory as sg
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from libqoi import mix_and_split, augment
from liberr import mean_relerr

# Fixing the seed
torch.manual_seed(0)

# Number of time series
n_samples = 500
# Length of each time series
n_times = 20
# Depth of the signature truncation
sig_depth = 5
sig_len = (2 ** (sig_depth + 1)) - 2


###
### PART 1: constructing the dataset
###
x_tmp = torch.normal(0., 3., (n_samples, n_times, 1))
y_data = torch.zeros(n_samples, 1)

for nth in range(n_samples):
    # Set to zero the first value, because we decide to construct data so
    x_tmp[nth][0] = 0.
    # Build the labels as the results of applying the max function
    y_data[nth] = torch.max(x_tmp[nth])

x_tmp2 = augment(x_tmp)
x_data = sg.signature(x_tmp2, sig_depth)

x_train, y_train, x_val, y_val = mix_and_split(x_data, y_data)


###
### PART 2: building the models, chosen both to be linear
###
# Setting Linear models for both the cases
model = nn.Linear(sig_len, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# And finally the loss functions, canonical
loss_fn = nn.MSELoss()

n_epochs = 10000
# History of loss function values
hist_loss_train = torch.ones(n_epochs)
hist_loss_val = torch.ones(n_epochs)

# Training Loop for raw data
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

# Now I need to measure the QUALITY of the training
# For instance we can plot true y VS predicted y
# ...maybe also compute the mean relative error?

# Store the overall predicted results
y_pred = model(x_data).detach()

# Error estimation
mre = mean_relerr(y_data, y_pred, 1e-2)[0]
print(f"Mean Relative Error: {mre:.1}%")
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
plt.title(f"Approximating MAX using Lin(Sig) [mre = {mre:.1f}%]")
plt.legend()
plt.grid()
plt.show()
