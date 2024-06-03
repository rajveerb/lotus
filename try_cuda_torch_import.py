import torch
import torchvision
from torchvision import datasets, transforms

# creta an empty tensor
x = torch.empty(5, 3)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_dataset = datasets.ImageFolder(
    "/mydata/imagenet",
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],	
    ),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=1,
    pin_memory=False,
)

for i, (input, target) in enumerate(train_loader):
    print(i)
    # print cuda memory
    print(torch.cuda.memory_allocated())
    # print cuda memory cache
    print(torch.cuda.memory_reserved())
    # print total cuda memory
    print(torch.cuda.max_memory_allocated())

