# Language Detection

Detect the language of short or long text inputs using machine learning and heuristics. This repository contains code, models, tests, and utilities for building, evaluating, and deploying language detection systems for many languages.

## Table of contents

- [About](#about)
- [Features](#features)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Quick start](#quick-start)
- [Usage](#usage)
  - [Command line](#command-line)
  - [Python API](#python-api)
  - [REST API / Docker](#rest-api--docker)
- [Models & Data](#models--data)
- [Evaluation](#evaluation)
- [Development & Contribution](#development--contribution)
  - [Code structure](#code-structure)
  - [Running tests](#running-tests)
  - [How to contribute](#how-to-contribute)
- [Performance & Limitations](#performance--limitations)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## About

This project implements language identification (LID) — determining the language of a given text snippet. It supports multiple detection strategies (n-gram models, character-level neural networks, pretrained transformers) and provides utilities for training new models, evaluating detection accuracy, and serving models for production use.

## Features

- Fast, accurate language detection for short and long texts
- Multiple model options: rule-based, logistic regression on character n-grams, RNN/CNN classifiers, and transformer-based models
- Pretrained models for common languages (English, Spanish, French, German, Chinese, Arabic, Russian, etc.)
- Easy-to-use Python API and CLI
- Dockerized REST API for production deployment
- Tools for dataset creation, augmentation, and evaluation
- Unit tests and CI-ready configuration

## Getting started

### Requirements

- Python 3.8+ (3.10 recommended)
- pip
- Optional: GPU + CUDA for training deep models
- Docker (optional, for containerized serving)

Recommended virtual environment:
- venv or conda

### Installation

Clone the repo and install dependencies:

```
git clone https://github.com/yourusername/language-detection.git
cd language-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you only need inference and not training, install the lightweight extras:

```
pip install -r requirements-inference.txt
```

### Quick start

Run the bundled CLI detector on an example sentence:

```
python -m lid.cli detect "Bonjour, comment ça va?"
# Output: fr (French) — confidence: 0.98
```

Or use the Python API:

```python
from lid import Detector

detector = Detector.load_pretrained('compact-v1')
lang, conf = detector.predict("¿Cómo estás?")
print(lang, conf)   # es 0.97
```

## Usage

### Command line

Common CLI commands (replace with actual script names if different):

- Detect language:
  ```
  lid detect "This is a test sentence."
  ```
- Batch detect from file (one sentence per line):
  ```
  lid detect-file --input samples.txt --output results.csv --model compact-v1
  ```
- Train new model:
  ```
  lid train --config configs/compact.yaml
  ```

Use `lid --help` to see all options.

### Python API

Basic usage:

```python
from lid import Detector

# Load pretrained model
detector = Detector.load_pretrained('compact-v1')

# Single prediction
lang, conf = detector.predict("Ciao, come stai?")

# Batch prediction
sentences = ["Hello", "Bonjour", "Hola"]
preds = detector.predict_batch(sentences)
```

Training example (high-level):

```python
from lid import Trainer, Dataset

dataset = Dataset.from_folder('data/train')
trainer = Trainer(config='configs/compact.yaml')
trainer.train(dataset, output_dir='models/compact-v1')
```

### REST API / Docker

Start a containerized server (example):

```
docker build -t lid-server .
docker run -p 8000:8000 -e MODEL_PATH=/models/compact-v1 lid-server
```

Then request:

```
curl -X POST "http://localhost:8000/predict" -d '{"text":"Guten Tag"}'
# {"language": "de", "confidence": 0.99}
```

(See docker/ and api/ for implementation details.)

## Models & Data

- models/
  - compact-v1 — small and fast, good for short texts
  - large-transformer — higher accuracy, supports more languages
- datasets/
  - commoncrawl-derived corpus (preprocessed)
  - langid.py datasets
  - train/val/test splits included for reproducibility

If using external datasets, cite and follow their licenses.

## Evaluation

Evaluation scripts compute precision, recall, F1 and confusion matrices. Example:

```
python -m lid.eval --pred predictions.csv --gold gold.csv --metrics f1,confusion
```

Included benchmark results (example):

- compact-v1: accuracy 93.7% on test set (short sentences)
- transformer-large: accuracy 97.4% on test set (mixed lengths)

Re-run evaluation when you train new models. Use stratified sampling by language to avoid class imbalance issues.

## Development & Contribution

### Code structure

- lid/ — core library
  - models/ — model definitions and wrappers
  - data/ — dataset loading and preprocessing
  - train/ — training utilities
  - api/ — REST API
  - cli.py — command-line interface
- configs/ — example configs for training and inference
- scripts/ — helper scripts (data prep, conversion)
- tests/ — unit and integration tests

### Running tests

Run unit tests with pytest:

```
pip install -r requirements-dev.txt
pytest
```

Linting and formatting:

```
pre-commit install
black .
flake8
```

### How to contribute

- Fork the repo and create a feature branch
- Write tests for new features or bug fixes
- Keep changes small and focused
- Submit a pull request with a clear description and motivation
- Follow the coding standards in CONTRIBUTING.md

See CONTRIBUTING.md for more details.

## Performance & Limitations

- Very short strings (1–2 characters) are inherently ambiguous; confidence will be low.
- Dialects and closely related languages (e.g., Serbian/Croatian/Bosnian) can be confused.
- Mixed-language or code-switched text may require specialized models — consider segmenting text before detection.
- Transformer models are more accurate but require more memory/compute.

Tips to improve accuracy:
- Provide at least 5–10 words when possible.
- Normalize punctuation and remove noisy tokens (URLs, emojis) for short-text scenarios.
- Use ensemble of models for high-stakes applications.
