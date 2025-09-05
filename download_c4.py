from datasets import load_dataset
import os

# Define a path on your shared storage
save_path = "./datasets/c4_en"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print(f"Downloading C4 dataset to {save_path}...")

# Load the dataset (this requires internet)
# We only need a subset for train and validation as in the original script
# so we can use .select() to make the download smaller and faster.
#train_data = load_dataset('allenai/c4', 'en', split='train', streaming=True)
val_data = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

# Take the same number of samples as in your get_c4 function
#train_subset = train_data.take(50000)
val_subset = val_data.take(1200)

# Convert from iterable to a Dataset object
from datasets import Dataset
#train_dataset = Dataset.from_generator(lambda: (yield from train_subset))
val_dataset = Dataset.from_generator(lambda: (yield from val_subset))

# Save them to disk
#train_dataset.save_to_disk(os.path.join(save_path, 'train'))
val_dataset.save_to_disk(os.path.join(save_path, 'validation'))

print("Dataset saved successfully.")