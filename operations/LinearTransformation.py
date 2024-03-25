import torchvision.transforms as transforms
from PIL import Image
import time
import itt
import torch

Image.MAX_IMAGE_PIXELS = 1000000000


batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    image = Image.open(image_file)
    image = image.convert('RGB')
    resize_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    tensor_image = resize_transform(image)
    
    transformation_matrix = transformation_matrix = torch.eye(3 * 32 * 32)
    
    mean_vector = torch.full((3 * 32 * 32,), 0.1)
    
    linear_transformation = transforms.LinearTransformation(transformation_matrix, mean_vector)

    time.sleep(1)
    
    if i == 4:
        itt.resume()
    
    transformed_tensor = linear_transformation(tensor_image)
    
    if i == 4:
        itt.detach()