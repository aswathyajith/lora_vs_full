This repo analyzes differences between the generative behaviors of full fine-tuning and LoRA fine-tuning.

## Models
We perform analysis on the following models: 
- 

## Steps to Reproduce: 

### 1. Setup TogetherAI for fine-tuning models
Export API key: `export TOGETHER_API_KEY=xxxxx`

### 2. Install environment
```
    $ conda env create -f environment.yml
    $ conda activate gen_behavior_analysis
    $ pip install -e .
```

### 3. Data Prep
python src/process_datasets.py

### 4. Fine-tune models
Set paths to train and eval datasets 
```
    $ export TRAIN_DATA_PATH=data/processed_datasets/math/train/processed_dataset_packed.parquet
    $ export VAL_DATA_PATH=data/processed_datasets/math/val/processed_dataset_packed.parquet
    $ export MODEL_SUFFIX=_aswathy_N_2e7
```

`$ together fine-tuning create --training-file $TRAIN_DATA_PATH --validation-file $VAL_DATA_PATH \
--suffix `