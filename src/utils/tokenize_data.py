import argparse
import logging
import random
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Union

from datasets import Dataset, load_dataset, IterableDataset, load_from_disk  # type: ignore
from transformers import (  # type: ignore
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)


# see default of ignore_index
# for https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
LOSS_IGNORE_INDEX = -100

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def tokenize_variable_length(
    data: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
) -> BatchEncoding:
    tokenized = tokenizer(
        data["text"], add_special_tokens=add_special_tokens, truncation=False
    )
    return tokenized


def tokenize_constant_length(
    data: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    add_special_tokens: bool = True,
    add_labels: bool = True,
) -> BatchEncoding:
    # tokenized contains `input_ids` and `attention_mask`
    tokenized: BatchEncoding = tokenizer(
        data["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=add_special_tokens,
    )
    # add labels to mask out any padding tokens
    if add_labels:
        tokenized["labels"] = [
            LOSS_IGNORE_INDEX if token_id == tokenizer.pad_token_id else token_id
            for token_id in tokenized["input_ids"]
        ]

    return tokenized


def pack_sequences(
    batch: BatchEncoding,
    max_seq_len: int,
    pad_token_id: int,
    eos_token_id: int,
    add_labels: bool,
    cutoff_size: int = 0,
) -> Dict[str, List[List[int]]]:
    """
    cutoff_size = max_seq_len means that we will drop any non-full sequences
        (full packing without padding)
    Example:
        Sequence 1:
        ['<s>', '▁usually', '▁,', '▁he', '▁would', '▁be', '▁t', 'earing']
        Sequence 2:
        ['▁around', '▁the', '▁living', '▁room', '▁,', '▁playing', '▁with', '▁his']
        Sequence 3:
        ['▁toys', '▁.', '</s>', '<s>', '▁but', '▁just', '▁one', '▁look']
    """
    packed_sequences = []
    buffer = []

    for input_ids in batch["input_ids"]:
        # Add the current sequence to the buffer
        buffer.extend(input_ids)
        buffer.append(eos_token_id)  # Add EOS at the end of each sequence

        # Check if buffer needs to be split into chunks
        while len(buffer) > max_seq_len:
            # Take a full chunk from the buffer and append it to packed_sequences
            packed_sequences.append(buffer[:max_seq_len])
            # Remove the processed chunk from the buffer
            buffer = buffer[max_seq_len:]

    # Add the last buffer if it's exactly chunk_size
    if len(buffer) == max_seq_len:
        packed_sequences.append(buffer)
    elif len(buffer) > cutoff_size:
        # if the buffer is larger than the cutoff size, pad it to the chunk_size
        # if not, we do not include in the packed_sequences
        buffer.extend([pad_token_id] * (max_seq_len - len(buffer)))
        packed_sequences.append(buffer)

    output = {"input_ids": packed_sequences}
    if add_labels:
        output["labels"] = [
            [
                LOSS_IGNORE_INDEX if token_id == pad_token_id else token_id
                for token_id in example
            ]
            for example in output["input_ids"]
        ]

    # mask attention for padding tokens, a better version would also mask cross-sequence dependencies
    output["attention_mask"] = [
        [0 if token_id == pad_token_id else 1 for token_id in example]
        for example in output["input_ids"]
    ]
    return output


def process_fast_packing(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    add_labels: bool,
    add_special_tokens: bool,
) -> Dataset:
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_variable_length(
            examples, tokenizer, add_special_tokens=add_special_tokens
        ),
        batched=True,
        num_proc=cpu_count(),
        load_from_cache_file=True,
        remove_columns=dataset.column_names,
    )
    logger.info(f"tokenized dataset: {tokenized_dataset}")

    packed_dataset = tokenized_dataset.map(
        lambda batch: pack_sequences(
            batch,
            max_sequence_length,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            add_labels=add_labels,
            cutoff_size=max_sequence_length,
        ),
        batched=True,
        num_proc=cpu_count() if len(tokenized_dataset) > 10000 else 1,
        remove_columns=["attention_mask"],
    )
    logger.info(f"Packed dataset: {packed_dataset}")
    return packed_dataset


def process_data(args: argparse.Namespace) -> None:
    if not args.out_filename.endswith(".parquet"):
        raise ValueError("`--out_filename` should have the `.parquet` extension")

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    dataset.to_json("dataset.jsonl", orient="records", lines=True)

    if not args.packing:
        tokenized_data = dataset.map(
            partial(
                tokenize_constant_length,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                add_special_tokens=False,
                add_labels=args.add_labels,
            ),
            batched=False,
            num_proc=cpu_count(),
            remove_columns=dataset.column_names,
        )
    else:
        tokenized_data = process_fast_packing(
            dataset,
            tokenizer,
            max_sequence_length=args.max_seq_length,
            add_labels=args.add_labels,
            add_special_tokens=True,
        )

    assert (
        "input_ids" in tokenized_data.column_names
        and "attention_mask" in tokenized_data.column_names
    )

    if args.add_labels:
        assert "labels" in tokenized_data.column_names

    logger.info("Tokenized data:")
    print(tokenized_data)

    logger.info(f"Saving data to {args.out_filename}")
    print(len(tokenized_data[0]["input_ids"]))
    tokenized_data.to_parquet(args.out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretokenize examples for finetuning via Together"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="clam004/antihallucination_dataset",
        help="Dataset name on the Hugging Face Hub",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="clam004/antihallucination_dataset",
        help="Dataset name on the Hugging Face Hub",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--add-labels",
        action="store_true",
        help="Whether to add loss labels from padding tokens",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name (for example, togethercomputer/Llama-3-8b-hf)",
    )
    parser.add_argument(
        "--out-filename",
        default="processed_dataset.parquet",
        help="Name of the Parquet file to save (should have .parquet extension)",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Whether to pack shorter sequences up to `--max-seq-length`",
    )
    args = parser.parse_args()

    process_data(args)


# python src/utils/tokenize_data.py --tokenizer="google/gemma-3-1b-pt" --max-seq-length=32768 --add-labels --packing --out-filename="processed_dataset_packed.parquet"


def tokenize_dataset(
    dataset: Union[Dataset, IterableDataset, str],
    tokenizer: Union[PreTrainedTokenizerBase, str],
    text_column: str = "text",
    max_length: int = 2048,
    target_tokens: Optional[int] = None,
    streaming_threshold: int = 10000,
    shuffle_seed: Optional[int] = 42,
    add_special_tokens: bool = True,
    truncation: bool = True,
    padding: Union[bool, str] = "max_length",
    add_labels: bool = False,
    num_proc: Optional[int] = None,
    **kwargs
) -> Union[Dataset, IterableDataset]:
    """
    Tokenize a Hugging Face dataset with optional streaming and shuffling for large datasets.
    
    Args:
        dataset: Hugging Face dataset (Dataset, IterableDataset, or dataset name string)
        tokenizer: Tokenizer instance or tokenizer name string
        text_column: Name of the text column to tokenize
        max_length: Maximum sequence length for tokenization
        target_tokens: Target number of tokens for subset creation (if None, uses full dataset)
        streaming_threshold: Dataset size threshold above which to use streaming
        shuffle_seed: Random seed for shuffling (None for no shuffling)
        add_special_tokens: Whether to add special tokens
        truncation: Whether to truncate sequences
        padding: Padding strategy ("max_length", True, False)
        add_labels: Whether to add labels for language modeling
        num_proc: Number of processes for parallel processing (None for auto)
        **kwargs: Additional arguments passed to load_dataset
        
    Returns:
        Tokenized dataset (Dataset or IterableDataset)
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = tokenize_dataset(
        ...     "wikitext", 
        ...     tokenizer, 
        ...     target_tokens=1000000,
        ...     streaming_threshold=5000
        ... )
    """
    # Load tokenizer if string provided
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset if string provided
    if isinstance(dataset, str):
        # Determine if we should use streaming based on dataset size estimation
        use_streaming = kwargs.get('streaming', False)
        if not use_streaming and target_tokens is not None:
            # For subset creation, we'll use streaming to avoid loading full dataset
            use_streaming = True
            kwargs['streaming'] = True
        
        dataset = load_dataset(dataset, **kwargs)
        if isinstance(dataset, dict):
            dataset = dataset['train']  # Default to train split
    
    # Determine processing strategy
    use_streaming = isinstance(dataset, IterableDataset) or (
        hasattr(dataset, '__len__') and len(dataset) > streaming_threshold
    )
    
    # Set up tokenization function
    def tokenize_function(examples):
        """Tokenize text examples."""
        tokenized = tokenizer(
            examples[text_column],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )
        
        # Add labels for language modeling if requested
        if add_labels:
            if padding == "max_length" or padding is True:
                # Create labels, masking padding tokens
                tokenized["labels"] = [
                    [LOSS_IGNORE_INDEX if token_id == tokenizer.pad_token_id else token_id 
                     for token_id in input_ids]
                    for input_ids in tokenized["input_ids"]
                ]
            else:
                # For variable length, labels are same as input_ids
                tokenized["labels"] = tokenized["input_ids"]
        
        return tokenized
    
    # Process dataset based on size and requirements
    if target_tokens is not None and use_streaming:
        # Use streaming for subset creation
        logger.info(f"Creating subset with target {target_tokens} tokens using streaming")
        
        # Shuffle the dataset if seed provided
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)
        
        # Tokenize and collect examples until we reach target tokens
        tokenized_examples = []
        total_tokens = 0
        
        for example in dataset:
            # Tokenize single example
            tokenized = tokenize_function({text_column: [example[text_column]]})
            example_tokens = len(tokenized["input_ids"][0])
            
            # Check if adding this example would exceed target
            if total_tokens + example_tokens > target_tokens and total_tokens > 0:
                break
            
            # Add to collection
            for key in tokenized:
                if key not in tokenized_examples:
                    tokenized_examples[key] = []
                tokenized_examples[key].append(tokenized[key][0])
            
            total_tokens += example_tokens
            
            # Log progress
            if len(tokenized_examples["input_ids"]) % 1000 == 0:
                logger.info(f"Processed {len(tokenized_examples['input_ids'])} examples, "
                          f"total tokens: {total_tokens}")
        
        # Convert to Dataset
        from datasets import Dataset
        tokenized_dataset = Dataset.from_dict(tokenized_examples)
        logger.info(f"Created subset with {len(tokenized_dataset)} examples, "
                   f"total tokens: {total_tokens}")
        
    else:
        # Standard processing for smaller datasets or full dataset
        if use_streaming:
            logger.info("Using streaming for large dataset")
            # For streaming datasets, we can't easily determine length
            # Process in batches
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None,
            )
        else:
            # Regular dataset processing
            logger.info(f"Processing dataset with {len(dataset)} examples")
            
            # Shuffle if requested
            if shuffle_seed is not None:
                dataset = dataset.shuffle(seed=shuffle_seed)
            
            # Set number of processes
            if num_proc is None:
                num_proc = cpu_count() if len(dataset) > 1000 else 1
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=dataset.column_names,
                load_from_cache_file=True,
            )
    
    logger.info(f"Tokenization complete. Dataset features: {tokenized_dataset.features}")
    return tokenized_dataset