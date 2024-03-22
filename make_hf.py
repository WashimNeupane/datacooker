from datasets import load_dataset
from tqdm import tqdm

hf_data_culturX = "uonlp/CulturaX"
hf_data_madlad = "allenai/MADLAD-400"
language_target = "en"

print("Downloading CulturaX")
dataset = load_dataset(hf_data_culturX, subset="en")

# Calculate total size of CulturaX dataset
total_culturX_size = sum(len(dataset[split]) for split in dataset)

# Iterate over dataset splits
for split, split_dataset in dataset.items():
    with tqdm(total=len(split_dataset), desc=f'Downloading {split}') as pbar:
        for data in split_dataset:
            # Process each data point
            split_dataset.to_json(f"culturX-{split}.jsonl")
            # Update progress bar
            pbar.update(1)

print("CulturaX download finished")

print("Downloading MADLAD dataset")
dataset = load_dataset(hf_data_madlad, language_target, split="clean")

# Calculate total size of MADLAD dataset
total_madlad_size = sum(len(dataset[split]) for split in dataset)

# Iterate over dataset splits
for split, split_dataset in dataset.items():
    with tqdm(total=len(split_dataset), desc=f'Downloading {split}') as pbar:
        for data in split_dataset:
            # Process each data point
            split_dataset.to_json(f"madlad-{split}.jsonl")
            # Update progress bar
            pbar.update(1)

print("MADLAD download finished")
