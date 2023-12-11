import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the text file
with open('output', 'r') as file:
    data = file.readlines()

# Create lists to store the parsed data
iterations = []
crop_times = []
flip_times = []
avg_crop_times = []
avg_flip_times = []
batch_forward_times = []
batch_backward_times = []

# Parse the data and populate the lists
for line in data:
    if "Time taken by randomResizedCrop" in line:
        _, crop_time, _, flip_time, _ = line.split()[-5:]
        crop_times.append(float(crop_time))
        flip_times.append(float(flip_time))
    elif "Average time taken by randomResizedCrop" in line:
        _, avg_crop_time, _, avg_flip_time, _ = line.split()[-5:]
        avg_crop_times.append(float(avg_crop_time))
        avg_flip_times.append(float(avg_flip_time))
    elif "Time taken by forward pass" in line:
        _, forward_time, _, _, batch_size = line.split()[-5:]
        batch_forward_times.append(float(forward_time))
    elif "Time taken by backward pass" in line:
        _, backward_time, _, _, batch_size = line.split()[-5:]
        batch_backward_times.append(float(backward_time))

# Create a DataFrame
df = pd.DataFrame({
    'Iterations': iterations,
    'Crop Times': crop_times,
    'Flip Times': flip_times,
    'Avg Crop Times': avg_crop_times,
    'Avg Flip Times': avg_flip_times,
    'Batch Forward Times': batch_forward_times,
    'Batch Backward Times': batch_backward_times,
})

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Iterations'], df['Crop Times'], label='Crop Times')
plt.plot(df['Iterations'], df['Flip Times'], label='Flip Times')
plt.plot(df['Iterations'], df['Avg Crop Times'], label='Avg Crop Times')
plt.plot(df['Iterations'], df['Avg Flip Times'], label='Avg Flip Times')
plt.plot(df['Iterations'], df['Batch Forward Times'], label='Batch Forward Times')
plt.plot(df['Iterations'], df['Batch Backward Times'], label='Batch Backward Times')

plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('Time Measurements Over Iterations')
plt.legend()
plt.show()
