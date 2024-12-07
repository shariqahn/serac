from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("locuslab/TOFU","full")
    print(dataset["train"][0])
    print(dataset["train"][1])
    print(dataset["train"][2])
    print(dataset["train"].column_names)
    print(dataset["validation"].column_names)
    print(dataset["test"].column_names)