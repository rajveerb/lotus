import time
from torchvision import transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

num_iterations = 100

crop_transform = transforms.FiveCrop(32)

input_image = Image.open("random_image10MB.jpg")

start_time = time.time()

for _ in range(num_iterations):
    transformed_images = crop_transform(input_image)

end_time = time.time()
elapsed_time = end_time - start_time

average_time = elapsed_time / num_iterations

print(f"Elapsed Time: {elapsed_time:.4f} seconds")
print(f"Average Time per Iteration: {average_time:.6f} seconds")