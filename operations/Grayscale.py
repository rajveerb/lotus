import torchvision.transforms as transforms
from PIL import Image
import time
import itt

# increase PIL image open size if necessary
Image.MAX_IMAGE_PIXELS = 1000000000

# Number of channels for the output grayscale image (1 or 3)
num_output_channels = 1  # For a single-channel grayscale image

grayscale_transform = transforms.Grayscale(num_output_channels=num_output_channels)

batch = ["random_image10MB.jpg" for i in range(5)]

for i, image_file in enumerate(batch):
    # Open the image
    image = Image.open(image_file)
    image = image.convert('RGB')
    
    # sleep for 1 sec to simulate some processing delay
    time.sleep(1)

    if i == 4:
        # Resume ITT if using Intel's ITT performance profiling
        itt.resume()

    # Apply Grayscale transformation to the image
    image = grayscale_transform(image)

    if i == 4:
        # Detach ITT if using Intel's ITT performance profiling
        itt.detach()