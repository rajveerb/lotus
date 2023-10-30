import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import StepLR

# An instance of your model.
model = torchvision.models.resnet18()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load("/mnt/c/Users/mayur/Downloads/checkpoint.pth.tar")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
model.load_state_dict(checkpoint["state_dict"], strict=False)
torch.save(model.module.state_dict(), 'non-parallel-model.pth')
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])


model_np = torchvision.models.resnet18()
model_np = model_np.cuda()
model_np.load_state_dict(torch.load('non-parallel-model.pth'), strict=False)
model_np = model_np.eval()

example = torch.rand(1, 3, 224, 224).cuda()
traced_script_module = torch.jit.trace(model_np, example)
traced_script_module.save("traced_resnet_model.pt")

# ------------------------

# model = AnyNet(args)
# model = nn.DataParallel(model).cuda()
# checkpoint = torch.load('/content/checkpoint/kitti2015_ck/checkpoint.tar')
# model.load_state_dict(checkpoint['state_dict'], strict=False)
# torch.save(model.module.state_dict(), 'non-parallel-model.pth')

# model_np = AnyNet(args)
# model_np = model_np.cuda()
# model_np.load_state_dict(torch.load('non-parallel-model.pth'), strict=False)
# model_np = model_np.eval()

# example = [torch.rand(1, 3, 1280, 720).cuda(), torch.rand(1, 3, 1280, 720).cuda()]
# traced_model = torch.jit.trace(model_np, example_inputs=example)
# traced_model.save("anynet_1280x720_b1.traced.pt")