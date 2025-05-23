import sys
sys.path.append('/home/rbachkaniwala3/work/XXXXBR-python')
import torchvision.transforms as transforms
from PIL import Image
import time,amdprofilecontrol as a
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["/home/rbachkaniwala3/work/XXXXBR-python/AMD/random_image10MB.jpg" for i in range(5)]

for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    # sleep for 1 sec
    time.sleep(1)

    if i == 4:
        a.resume(1)
        # normalize for imagetnet tpye
    image = transforms.ToTensor()(image)
    if i == 4:
        a.pause(1)