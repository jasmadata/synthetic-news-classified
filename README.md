---
license: apache-2.0
size_categories: n<1K
dataset_info:
  features:
  - name: text
    dtype: string
    description: The news article text content
  - name: label
    dtype:
      class_label:
        names:
          '0': science
          '1': technology
          '2': business
          '3': health
          '4': entertainment
          '5': environment
          '6': sports
          '7': politics
    description: The categorical label indicating the news category
  splits:
  - name: train
    num_bytes: 32631
    num_examples: 100
  download_size: 22141
  dataset_size: 32631
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- synthetic
- distilabel
- rlaif
- datacraft
- text-classification
- news-classification
---

# Synthetic News Classification Dataset

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset Size](https://img.shields.io/badge/Size-32.6KB-green.svg)]()
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/argilla/synthetic-news-classified)

A high-quality synthetic dataset for news article classification, created using [distilabel](https://distilabel.argilla.io/).

## Quick Start

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("argilla/synthetic-news-classified")

# Or load a specific configuration
dataset = load_dataset("argilla/synthetic-news-classified", "default")
```

## Dataset Overview

### Description
This dataset contains synthetically generated news articles across 8 different categories, designed for training and evaluating text classification models. Each example consists of a news article text and its corresponding category label.

### Features
- **text**: News article content (string)
- **label**: Category classification (integer)
  - 0: Science
  - 1: Technology
  - 2: Business
  - 3: Health
  - 4: Entertainment
  - 5: Environment
  - 6: Sports
  - 7: Politics

### Statistics
- **Total Examples**: 100
- **Dataset Size**: 32.6KB
- **Split**: Train only
- **Average Text Length**: ~300 words

## Generation Process

### Pipeline Reproduction
The dataset can be regenerated using the provided pipeline configuration:

### Example Structure

```json
{
    "label": 4,
    "text": "A star-studded cast, including Leonardo DiCaprio and Jennifer Lawrence, has been announced for the upcoming biographical drama film about the life of the famous musician, Elvis Presley. The movie, directed by Baz Luhrmann, is set to release in summer 2024 and promises to be a musical spectacle."
}
```

## Usage Examples

### Basic Loading and Exploration
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("argilla/synthetic-news-classified")

# Print basic information
print(f"Dataset size: {len(dataset['train'])} examples")
print(f"Features: {dataset['train'].features}")

# Display first example
print(dataset['train'][0])
```

### Data Processing
```python
# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

# Get label distribution
label_dist = df['label'].value_counts()
print("Label Distribution:")
print(label_dist)
```

### Model Training Example
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Prepare tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=8
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)
```

## Citation

```bibtex
@misc{synthetic-news-classified,
    title={Synthetic News Classification Dataset},
    author={Argilla},
    year={2024},
    publisher={Hugging Face},
}
```

## License
This dataset is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.



