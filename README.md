# Mini-BERT: A Lightweight BERT Implementation

A compact implementation of BERT (Bidirectional Encoder Representations from Transformers) for masked language modeling, trained on "The Adventures of Sherlock Holmes" by Arthur Conan Doyle.

## 📋 Overview

This project implements a minimal version of BERT using PyTorch, demonstrating the core concepts of transformer-based language models. The model learns to predict masked words in sentences by training on classic literature text.

## ✨ Features

- **Lightweight Architecture**: Compact transformer model with configurable depth and dimensions
- **Masked Language Modeling**: Trains using the MLM objective (predicting 15% masked tokens)
- **Custom Tokenizer**: Implements a strict tokenizer that removes punctuation and filters rare words
- **GPU Support**: Automatically utilizes CUDA if available
- **Training Visualization**: Generates loss curve plots to track training progress
- **Interactive Prediction**: Test the model with custom sentences containing [MASK] tokens

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd MiniBERT
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
MiniBERT/
├── main.py           # Main training and model implementation
├── data.txt          # Training corpus (Sherlock Holmes stories)
├── requirements.txt  # Python dependencies
├── loss.png         # Generated training loss plot (after training)
└── README.md        # This file
```

## 🏗️ Model Architecture

### Configuration

- **Sequence Length**: 32 tokens
- **Embedding Dimension**: 64
- **Attention Heads**: 4
- **Transformer Layers**: 2
- **Batch Size**: 32
- **Training Epochs**: 500
- **Learning Rate**: 1e-3
- **Masking Probability**: 15%

### Components

1. **Token Embedding**: Converts tokens to dense vector representations
2. **Position Encoding**: Adds positional information to embeddings
3. **Transformer Encoder**: Multi-head self-attention layers
4. **Output Layer**: Linear layer projecting to vocabulary size

## 🎯 Usage

### Training the Model

Simply run the main script:

```bash
python main.py
```

The script will:
1. Load and tokenize the text data
2. Build a vocabulary (filtering words appearing ≤1 time)
3. Initialize the Mini-BERT model
4. Train for 500 epochs
5. Save a loss curve plot as `loss.png`
6. Run example predictions

### Example Output

```
Loading Data...
Vocab Size: 4523 (Punctuation Removed)
Model Parameters: 328,987
Starting Training...
Epoch 10/500 | Loss: 5.4231
Epoch 20/500 | Loss: 4.8762
...
Saved loss.png

Input: Sherlock is a [MASK] detective
Predictions:
  great: 8.4523
  famous: 7.9234
  good: 7.2341
  brilliant: 6.8901
  private: 6.5432
```

### Making Custom Predictions

Use the `predict()` function in the code:

```python
predict(model, tokenizer, "The [MASK] sat on the mat")
predict(model, tokenizer, "Holmes is a brilliant [MASK]")
```

## 🔧 Customization

### Adjusting Model Size

Edit the `Config` class in [main.py](main.py):

```python
class Config:
    SEQ_LEN = 64          # Longer sequences
    EMBED_DIM = 128       # Larger embeddings
    N_HEADS = 8           # More attention heads
    N_LAYERS = 4          # Deeper network
    BATCH_SIZE = 16       # Smaller batches
    EPOCHS = 1000         # More training
    LR = 5e-4             # Adjusted learning rate
```

### Using Different Text Data

Replace [data.txt](data.txt) with your own text corpus. The model will automatically tokenize and train on the new data.

## 📊 Training Details

### Data Preprocessing

1. **Tokenization**: Text is split into words, converted to lowercase
2. **Punctuation Removal**: All punctuation is stripped to simplify vocabulary
3. **Vocabulary Filtering**: Rare words (frequency ≤1) are replaced with `[UNK]`
4. **Sliding Windows**: Creates overlapping sequences of fixed length

### Special Tokens

- `[PAD]`: Padding token (ID: 0)
- `[MASK]`: Mask token for MLM (ID: 1)
- `[UNK]`: Unknown token for out-of-vocabulary words (ID: 2)

### Training Strategy

- **Objective**: Cross-entropy loss on masked token predictions
- **Masking**: Randomly masks 15% of tokens (excluding special tokens)
- **Optimizer**: Adam with default parameters
- **Loss Calculation**: Only computed for masked positions (unmasked targets set to -100)

## 🛠️ Requirements

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Loss visualization
- **CUDA**: Optional GPU acceleration (via PyTorch)

See [requirements.txt](requirements.txt) for specific versions.

## 📈 Performance

The model is intentionally small for educational purposes and fast training. Expected behavior:

- **Training Time**: ~10-30 minutes on CPU (faster on GPU)
- **Loss Convergence**: Typically reaches ~2-3 loss after 500 epochs
- **Prediction Quality**: Learns common word associations from the training corpus

## 🎓 Educational Purpose

This project is designed to demonstrate:

- Transformer architecture fundamentals
- Masked language modeling (MLM) training
- Self-attention mechanisms
- Tokenization and vocabulary building
- PyTorch model implementation
- Training loop best practices

## 🤝 Contributing

Feel free to:
- Experiment with different hyperparameters
- Try alternative training corpora
- Implement additional BERT variants (NSP, SOP)
- Add evaluation metrics and benchmarks

## 📝 License

This project uses public domain text from Project Gutenberg. The code is available for educational and research purposes.

## 🙏 Acknowledgments

- **Text Data**: "The Adventures of Sherlock Holmes" by Arthur Conan Doyle (Project Gutenberg)
- **Inspiration**: Original BERT paper by Devlin et al. (2018)
- **Framework**: PyTorch team for the excellent deep learning library

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This is a minimal implementation for learning purposes. For production use cases, consider using pre-trained models from Hugging Face Transformers library.
