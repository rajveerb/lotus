import torch
import torch.nn as nn
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

checkpoint = torch.load("checkpoint.pth.tar")
state_dict = checkpoint["state_dict"]
state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)

model = model.module if isinstance(model, nn.DataParallel) else model

model.eval()

example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("traced_resnet_model.pt")
