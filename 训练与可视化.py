import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from 数据预处理 import dataClean
from LSTM搭建 import LSTM

dataclean = dataClean()
model = LSTM()
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for t in range(num_epochs):
    y_train_pred = model(dataClean.x_train)

    loss = criterion(y_train_pred, dataclean.y_train)
    print("Epoch ", t, "MSE: ", loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# training_time = time.time() - start_time
# print("Training time: {}".format(training_time))

predict = pd.DataFrame(dataclean.scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(dataclean.scaler.inverse_transform(dataclean.y_train.detach().numpy()))

sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 1, 1)
ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (LSTM)", color='tomato')
ax.set_title('Stock price', size=14, fontweight='bold')
ax.set_xlabel("Days", size=14)
ax.set_ylabel("Cost (USD)", size=14)
ax.set_xticklabels('', size=10)
plt.show()
