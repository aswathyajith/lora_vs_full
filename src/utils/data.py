from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

def download_dataset(
    hf_dataset_name: str, 
    data_dir: str = None, 
    download_path: str = "/net/projects/clab/aswathy/projects/ft_analysis/data/downloaded_datasets/test", 
    streaming: bool = False, 
    buffer_size: int = 10000, 
    n_samples: int = None
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
        dataset = shuffled_dataset.take(n_samples)
        
    # download to path 
    dataset.save_to_disk(download_path)
    print(f"Dataset saved to {download_path}")
    return dataset

def tokenize_dataset(
    dataset: Dataset | DatasetDict, 
    model: str, 
    text_column: str = "text"
) -> Dataset | DatasetDict:
    """Tokenize a Dataset (or DatasetDict) for a model (no padding or truncation)"""
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    return dataset.map(lambda example: tokenize(tokenizer, example, text_column), batched=True)

def tokenize(
    tokenizer: AutoTokenizer, 
    example: dict, 
    text_column: str = "text"
) -> dict: 
    return tokenizer(example[text_column])

def check_dataset_size(dataset):
    """Check the size of a dataset in GB"""
    size = dataset.info.download_size / (1024 ** 3)
    return size

def stream_dataset(dataset: Dataset | DatasetDict, batch_size: int = 1000) -> Generator[dict, None, None]:
    """Stream a dataset"""
    for batch in dataset.iter(batch_size=batch_size):
        yield batch