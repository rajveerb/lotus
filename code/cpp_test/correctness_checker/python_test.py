import os
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Specify the directory paths
testdir = 'imagenet/test/images2'  # Replace with your test directory path
output_file = 'processed_test_dataset.txt'  # Output text file to store processed dataset
normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
# Define the transformations
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(64, (1,1), (1,1)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    normalize
    ])

# Create an empty list to store processed dataset

# Iterate through the images in the test directory
for root, dirs, files in os.walk(testdir):
    for filename in files:
        # Check if the file is an image (you can add more image extensions as needed)
        image_path = os.path.join(root, filename)
        
        # Load and apply transformations to the image
        image = Image.open(image_path)

        processed_image = data_transforms(image)

        np.savetxt(output_file, processed_image.reshape((3,-1)), fmt="%s", header=str(processed_image.shape))

