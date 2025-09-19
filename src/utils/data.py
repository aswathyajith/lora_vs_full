from datasets import IterableDatasetDict, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

def download_dataset(
    hf_dataset_name: str, 
    data_dir: str = None, 
    download_path: str = "/net/projects/clab/aswathy/projects/ft_analysis/data/downloaded_datasets/tmp", 
    streaming: bool = False, 
    buffer_size: int = 10000, 
    n_samples: int = 10000
) -> Dataset | DatasetDict:
    """Download a dataset from the Hugging Face Hub and save it to the cache directory"""
    dataset = load_dataset(
        hf_dataset_name, 
        data_dir, 
        cache_dir=download_path, 
        streaming=streaming
    )
    # Shuffle the dataset when streaming is enabled
    # Streaming is usually enabled when the dataset 
    # is too large to be downloaded and fit in memory

    if streaming: 
        print(f"Shuffling dataset with streaming buffer size {buffer_size}")
        shuffled_dataset = dataset.shuffle(buffer_size=buffer_size)
        print(f"Taking {n_samples} samples from the shuffled dataset")
        if type(shuffled_dataset) == IterableDatasetDict:
            splits = list(shuffled_dataset.keys())
            dataset = DatasetDict()
            print(f"Splits: {splits}")
            for split in splits:
                # convert iterable dataset to dataset
                dataset[split] = Dataset.from_list(list(shuffled_dataset[split].take(n_samples)))
        else:
            dataset = list(shuffled_dataset.take(n_samples))
        
    # download to path 
    print(type(dataset))
    dataset.save_to_disk(download_path)
    print(f"Dataset saved to {download_path}")
    return dataset

def get_samples_with_target_tokens(
    dataset: Dataset, 
    n_tkns: int
) -> Dataset | DatasetDict:
    """Get samples with target number of tokens"""
    total_tokens = 0
    n_samples = 0
    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    for example in dataset:
        total_tokens += len(example["input_ids"])
        n_samples += 1
        if total_tokens >= n_tkns:
            break
    
    print(f"Selected {n_samples} samples to reach {n_tkns} tokens")
    return dataset.select(range(n_samples))

def tokenize_dataset(
    dataset: Dataset | DatasetDict, 
    model: str, 
    text_column: str = "text", 
    n_tkns_train: int = None, 
    n_tkns_val: int = None
) -> Dataset | DatasetDict:
    """Tokenize a Dataset (or DatasetDict) for a model (no padding or truncation)"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(lambda example: tokenize(tokenizer, example, text_column), batched=True)
    if text_column != "text":
        tokenized_dataset = tokenized_dataset.rename_column(text_column, "text")

    if n_tkns_train is not None:
        tokenized_dataset["train"] = get_samples_with_target_tokens(tokenized_dataset["train"], n_tkns_train)

    if n_tkns_val is not None:
        tokenized_dataset["val"] = get_samples_with_target_tokens(tokenized_dataset["val"], n_tkns_val)
        
    return tokenized_dataset

def tokenize(
    tokenizer: AutoTokenizer, 
    example: dict, 
    text_column: str = "text"
) -> dict: 
    return tokenizer(example[text_column])

def count_tokens(dataset: Dataset | DatasetDict, model: str, text_column: str = "text") -> int:
    """Count the number of tokens in a dataset"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    # apply tokenizer to dataset
    tokenized_data = dataset.map(lambda example: tokenize(tokenizer, example, text_column), batched=True)
    token_count = 0
    if type(tokenized_data) == DatasetDict:
        for ds_split in tokenized_data:
            for example in tokenized_data[ds_split]:
                token_count += len(example['input_ids'])
        return token_count
    else:
        for example in tokenized_data:
            token_count += len(example['input_ids'])
        return token_count

def create_train_test_splits(
    dataset: DatasetDict, 
    test_size: float = 0.2, 
    seed: int = 42, 
    path: str = "data/train_test_splits/tmp"
) -> None:
    """Create train and test splits for a dataset"""
    
    split_dataset = DatasetDict()
    # if single split or dataset object (not a DatasetDict), then split it into train and test
    if type(dataset) == Dataset or (type(dataset) == DatasetDict and len(dataset.keys()) < 2): 
            dataset_splits = list(dataset.keys())
            dataset = dataset[dataset_splits[0]] # get the first (and only) split, then split it into train and test
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            split_dataset["train"] = splits["train"]
            split_dataset["val"] = splits["test"]
            split_dataset.save_to_disk(path)
            print(f"Dataset saved to {path}")
    else: #DatasetDict with multiple splits
        
        
        existing_splits = dataset.keys()
        assert "train" in existing_splits, "Train split not found in dataset"

        
        split_dataset["train"] = dataset["train"]
        for split in existing_splits:
            if split != "train":
                split_dataset["val"] = dataset[split]
        split_dataset.save_to_disk(path)
        print(f"Dataset saved to {path}")




