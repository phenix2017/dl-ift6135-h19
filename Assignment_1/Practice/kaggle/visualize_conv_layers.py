import torch
import torchvision.utils as vutils

model = torch.load('model.pth')
model.load_state_dict(torch.load('model_epoch_0082_batch_00076_of_00282.pth'))

# Conv1
conv1_weights = model.conv1.weight.cpu().data
for i in range(len(conv1_weights)):
    conv1_weights[i] = (conv1_weights[i] - conv1_weights[i].min())/(conv1_weights[i].max() - conv1_weights[i].min())

vutils.save_image(conv1_weights, 'conv1_weights_init.png', nrow=10)
