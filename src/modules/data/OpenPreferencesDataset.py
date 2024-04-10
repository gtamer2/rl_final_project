from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("argilla/OpenHermesPreferences", split="train")

# Load a tokenizer and apply chat template
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
example = ds[0]
chosen_example = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
rejected_example = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
print(f"== Chosen example ==\n\n{chosen_example}")
print(f"== Rejected example ==\n\n{rejected_example}")
