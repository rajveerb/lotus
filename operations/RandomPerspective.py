import torchvision.transforms as transforms
from PIL import Image
import time
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

distortion_scale = 1  # This is an example value for distortion
p = 1  # Probability of applying the transformation
batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    random_perspective = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p)

    # sleep for 1 sec
    time.sleep(1)

    if i == 4:
        itt.resume()

    # Apply random perspective transformation to the image
    image = random_perspective(image)

    if i == 4:
        itt.detach()