import sys
sys.path.append('/home/rbachkaniwala3/work/rajveerb_AMDProfileControl-python')
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
import amdprofilecontrol as a
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["/data/imagenet/train/n03314780/n03314780_12371.JPEG" for i in range(128)]

batches = []
for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    image = transforms.RandomResizedCrop(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    batches.append(image)


a.resume(1)
# call default collate
batches = torch.utils.data.default_collate(batches)
a.pause(1)