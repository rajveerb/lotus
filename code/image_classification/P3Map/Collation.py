import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["code/image_classification/P3Map/random_image10MB.jpg" for i in range(1024)]

batches = []
for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    # image = transforms.RandomResizedCrop(224)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    batches.append(image)

itt.resume()
# call default collate
batches = torch.utils.data.default_collate(batches)
itt.detach()