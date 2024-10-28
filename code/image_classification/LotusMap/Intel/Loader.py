import torchvision.transforms as transforms
from PIL import Image
import time,itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["code/image_classification/LotusMap/Intel/random_image10MB.jpg" for i in range(5)]

for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
        # sleep for 1 sec
    time.sleep(1)
    # convert to RGB like torch's pil_loader
    if i == 4:
        itt.resume()
    # Below operation internally does the follows:
    # 1. Mem allocation (to store the image)
    # 2. Read file
    # 3. Decode file
    # 4. Copy the image (internally uses memcpy)

    image = image.convert('RGB')
    if i == 4:
        itt.detach()