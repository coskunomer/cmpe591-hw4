import torch
from torch.utils.data import DataLoader
from homework4 import CNP
from data_loader import CNMPDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# loading the data
dataset = CNMPDataset("cnmp_dataset.pkl")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30

epoch_losses = []

# training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for observation, target_x, target_y in dataloader:
        loss = model.nll_loss(observation.squeeze(0), target_x.squeeze(0), target_y.squeeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_losses.append(epoch_loss / len(dataloader))
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

model.eval()
end_eff_mse_list = []
obj_mse_list = []

for _ in range(1000):
    obs, tx, ty = dataset[random.randint(0, len(dataset) - 1)]
    with torch.no_grad():
        pred_mean, _ = model(obs, tx)
        pred = pred_mean.squeeze(0).numpy()
        gt = ty.squeeze(0).numpy()

    error = (pred - gt)
    end_eff_mse = np.mean(error[:, :2] ** 2)
    obj_mse = np.mean(error[:, 2:] ** 2)

    end_eff_mse_list.append(end_eff_mse)
    obj_mse_list.append(obj_mse)

# Plotting
means = [np.mean(end_eff_mse_list), np.mean(obj_mse_list)]
stds = [np.std(end_eff_mse_list), np.std(obj_mse_list)]

labels = ["End-effector", "Object"]
colors = ["skyblue", "salmon"]

fig, ax = plt.subplots(figsize=(8, 6))

x_pos = np.arange(len(labels)) 
ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
ax.set_title('CNMP Prediction Error Comparison', fontsize=16, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, EPOCHS+1), epoch_losses, marker='o', color='b', label='Training Loss')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Training Loss Curve', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()
