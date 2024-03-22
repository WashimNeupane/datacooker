from datasets import load_dataset

hf_data_culturX = "uonlp/CulturaX"
hf_data_madlad  = "allenai/MADLAD-400"
language_target = "en"

print("Downloading CulturX")
dataset = load_dataset(hf_data_culturX, subset="en")
for split, split_dataset in dataset.items():
    split_dataset.to_json(f"culturX-{split}.jsonl")

print("CulturX downlaod finished")
print("Downloading Madlad dataset")

dataset = load_dataset(hf_data_madlad, language_target, split="clean")
for split, split_dataset in dataset.items():
    split_dataset.to_json(f"culturX-{split}.jsonl")
