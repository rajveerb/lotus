import torchvision.transforms as transforms
from PIL import Image
import time,itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["/mydata/synthetic_data_10MBeach_8192files_symlink/train/class_1/symlink_1.jpg" for i in range(5)]

for i,image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    # convert to RGB like torch's pil_loader
    image = image.convert('RGB')
    # sleep for 1 sec
    time.sleep(1)

    if i == 4:
        itt.resume()
        # normalize for imagetnet tpye
    image = transforms.ToTensor()(image)
    if i == 4:
        itt.detach()