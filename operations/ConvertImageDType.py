import torchvision.transforms as transforms
from PIL import Image
import time
import itt
import torch

Image.MAX_IMAGE_PIXELS = 1000000000

convert_image_dtype = transforms.ConvertImageDtype(torch.float)

batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    tensor = transforms.functional.to_tensor(image)
    time.sleep(1)
    
    if i == 4:
        itt.resume()

    tensor = convert_image_dtype(tensor)

    if i == 4:
        itt.detach()