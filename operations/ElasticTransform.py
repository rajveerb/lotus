import torchvision.transforms as transforms
from PIL import Image
import time
import itt
# increase PIL image open size
Image.MAX_IMAGE_PIXELS = 1000000000

batch = ["random_image10MB.jpg" for i in range(5)]
elastic_transform = transforms.ElasticTransform()


for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    image_tensor = transforms.functional.to_tensor(image)

    time.sleep(1)

    if i == 4:
        itt.resume()

    image = elastic_transform(image_tensor)

    if i == 4:
        itt.detach()