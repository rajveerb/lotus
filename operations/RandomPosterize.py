import torchvision.transforms as transforms
from PIL import Image
import time
import itt

Image.MAX_IMAGE_PIXELS = 1000000000


batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    random_posterize_transform = transforms.RandomPosterize(bits=4, p=1)

    
    time.sleep(1)

    if i == 4:
        itt.resume()
    
    image = random_posterize_transform(image)

    if i == 4:
        itt.detach()