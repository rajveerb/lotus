import torchvision.transforms as transforms
from PIL import Image
import time
import itt
import torch

Image.MAX_IMAGE_PIXELS = 1000000000

to_pil_image = transforms.ToPILImage()

batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    image_tensor = transforms.ToTensor()(image)
    time.sleep(1)

    if i == 4:
        itt.resume()
        
    image = to_pil_image(image_tensor)

    if i == 4:
        itt.detach()