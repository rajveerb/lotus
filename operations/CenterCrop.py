import torchvision.transforms as transforms
from PIL import Image
import time
import itt

Image.MAX_IMAGE_PIXELS = 1000000000

size = (512, 512)
batch = ["random_image10MB.jpg" for i in range(5)]
center_crop = transforms.CenterCrop(size)

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    time.sleep(1)

    if i == 4:
        itt.resume()

    image = center_crop(image)

    if i == 4:
        itt.detach()