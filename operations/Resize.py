import torchvision.transforms as transforms
from PIL import Image
import time
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

interpolation = Image.BILINEAR
size = (256, 256)
batch = ["random_image10MB.jpg" for i in range(5)]

for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    resize_transform = transforms.Resize(size, interpolation)

    # sleep for 1 sec
    time.sleep(1)

    if i == 4:
        itt.resume()

    # Resize the image
    image = resize_transform(image)

    if i == 4:
        itt.detach()