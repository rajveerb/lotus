import torchvision.transforms as transforms
from PIL import Image
import time
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

p = 1  # Probability of the image being flipped (50% chance)
batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    # sleep for 1 sec
    time.sleep(1)
    vertical_flip = transforms.RandomVerticalFlip(p)


    if i == 4:
        itt.resume()

    # Apply random vertical flip to the image
    image = vertical_flip(image)

    if i == 4:
        itt.detach()