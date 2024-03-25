import torchvision.transforms as transforms
from PIL import Image
import time
import itt

Image.MAX_IMAGE_PIXELS = 1000000000

# Define your custom lambda function
# For example, let's invert the colors of the image
invert_colors = lambda x: 1 - x

lambda_transform = transforms.Lambda(invert_colors)

batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    image = transforms.functional.to_tensor(image)

    time.sleep(1)

    if i == 4:
        itt.resume()

    image = lambda_transform(image)

    if i == 4:
        itt.detach()