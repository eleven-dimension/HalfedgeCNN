import matplotlib.pyplot as plt
import os

losses = []
with open('packages/model/epoch_losses_resnet.txt', 'r') as file:
    for line in file:
        losses.append(float(line.strip()))

plt.figure(figsize=(10, 6))
plt.plot(losses, linestyle='-', color='b', label='moving average of data points')
plt.xlabel('data point index')
plt.ylabel('loss')
plt.legend()
plt.grid(False)


output_dir = 'packages/plot'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'training_loss_resnet_plot.png'))

plt.show()