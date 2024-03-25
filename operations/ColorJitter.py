import torchvision.transforms as transforms
from PIL import Image
import time
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

# Define parameters for ColorJitter
brightness = 0.2  
contrast = 0.2 
saturation = 0.2 
hue = 0.02        

color_jitter_transform = transforms.ColorJitter(
    brightness=brightness,
    contrast=contrast,
    saturation=saturation,
    hue=hue
)

batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    # sleep for 1 sec
    time.sleep(1)

    if i == 4:
        itt.resume()

    # Apply ColorJitter transformation to the image
    image = color_jitter_transform(image)

    if i == 4:
        itt.detach()