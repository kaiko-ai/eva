
import torch

from eva.vision.models.networks.encoders import PhikonEncoder


x = torch.Tensor(2, 3, 224, 224)

model = PhikonEncoder()

out = model(x)
# out = out.last_hidden_state[:, 1:, :]
print(out[0].shape)
