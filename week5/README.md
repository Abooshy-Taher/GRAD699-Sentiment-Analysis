# Week 5: Unsloth Fine-tuning for Sentiment Classification

## Overview

This week introduces **Unsloth-based LLM fine-tuning** for ternary sentiment classification using the same dataset and chronological split strategy as Week 4. The notebook fine-tunes a small LLaMA-family model (Llama-3.1-8B) using LoRA adapters and 4-bit quantization for efficient training.

## What is Unsloth?

**Unsloth** is a library that makes fine-tuning large language models (LLMs) faster and more memory-efficient. It provides:

- **2-5x faster training** compared to standard Hugging Face fine-tuning
- **Memory-efficient** training with 4-bit quantization and LoRA adapters
- **Easy-to-use API** for fine-tuning LLaMA, Mistral, Qwen, and other models
- **Colab-ready** with optimized configurations for T4 and A100 GPUs

## Key Features

- ✅ **Unsloth-based fine-tuning** with LoRA + 4-bit quantization
- ✅ **Chronological split** (same as Week 4) to prevent data leakage
- ✅ **Instruction-style SFT format** for LLM training
- ✅ **Colab-ready** with GPU checks and flexible data loading
- ✅ **Reproducible** with random seeds and split summaries

## Dataset

- **Source**: `Amazon_Data.csv`
- **Target**: Sentiment labels derived from ratings:
  - Rating ≤ 2 → Negative (label 0)
  - Rating = 3 → Neutral (label 1)
  - Rating ≥ 4 → Positive (label 2)

## Split Strategy

- **70% Train** (oldest data)
- **15% Validation** (middle)
- **15% Test** (most recent data)

**⚠️ CRITICAL**: No shuffling before split - maintains strict chronological order to prevent temporal leakage.

## Running in Google Colab

### Prerequisites

1. **GPU Required**: Unsloth fine-tuning requires a GPU (T4 or A100)
   - In Colab: Runtime → Change runtime type → GPU (T4 or A100)

2. **Dataset**: Upload `Amazon_Data.csv` to Google Drive
   - Option 1: Upload to `/content/drive/MyDrive/Amazon_Data.csv`
   - Option 2: Upload directly to `/content/Amazon_Data.csv`

### Quick Start

1. **Open the notebook in Colab**:
   
   **Note**: Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub username and repository name.
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/week5/week5_unsloth_sentiment.ipynb)
   
   Or upload the notebook directly to Colab and run it there.

2. **Run all cells sequentially**:
   - Cell 1: Install dependencies (takes ~2-3 minutes)
   - Cell 2: GPU check
   - Cell 3: Imports and configuration
   - Cell 4: Load and prepare data
   - Cell 5-9: Prepare dataset and configure model
   - Cell 10: Fine-tune model (takes 30-60 minutes)
   - Cell 11-14: Save model, evaluate, and show results

### Expected GPU Requirements

- **T4 GPU**: Works, but training may take longer (~60-90 minutes)
- **A100 GPU**: Recommended for faster training (~30-45 minutes)
- **CPU**: Not supported (will raise error)

### Training Configuration

The notebook uses conservative settings for memory efficiency:

- **Model**: Llama-3.1-8B (4-bit quantized)
- **LoRA rank**: 16
- **Batch size**: 2 per device (effective batch size: 8 with gradient accumulation)
- **Epochs**: 1 (can increase to 2-3 for better performance)
- **Learning rate**: 2e-4

### Adjusting Training Parameters

To improve performance, you can adjust:

1. **Increase epochs** (Cell 8):
   ```python
   num_train_epochs=2,  # Change from 1 to 2-3
   ```

2. **Increase LoRA rank** (Cell 6):
   ```python
   r=32,  # Change from 16 to 32 (more capacity, slower)
   ```

3. **Increase batch size** (Cell 8):
   ```python
   per_device_train_batch_size=4,  # If you have more GPU memory
   ```

4. **Full test evaluation** (Cell 13):
   ```python
   FULL_TEST_EVAL = True  # Evaluate on full test set (slower)
   ```

## File Structure

```
week5/
├── week5_unsloth_sentiment.ipynb   # Main notebook (Colab-ready)
├── week5_utils.py                  # Utility functions (optional)
└── README.md                        # This file
```

## Model Output

The fine-tuned model outputs sentiment labels as strings:
- `"Negative"` for ratings ≤ 2
- `"Neutral"` for rating = 3
- `"Positive"` for ratings ≥ 4

These are converted back to numeric labels (0, 1, 2) for metric computation.

## Evaluation Metrics

The notebook computes:
- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of predictions

## Reproducibility

- ✅ Random seeds set (`RANDOM_STATE=42`)
- ✅ Chronological split (no shuffling)
- ✅ Same label mapping as Week 4 (0=Negative, 1=Neutral, 2=Positive)

## Data Leakage Prevention

- ✅ Chronological split by timestamp
- ✅ No shuffling before split
- ✅ Test set used only once for final evaluation

## Troubleshooting

### "GPU required" error
- Enable GPU in Colab: Runtime → Change runtime type → GPU

### "Could not find Amazon_Data.csv"
- Upload the CSV file to Google Drive (`/content/drive/MyDrive/`) or `/content/`

### Out of memory errors
- Reduce batch size: `per_device_train_batch_size=1`
- Reduce LoRA rank: `r=8`
- Use a smaller model: `"unsloth/llama-3.1-1b-bnb-4bit"`

### Slow training
- Use A100 GPU instead of T4
- Reduce dataset size for testing (sample train/val sets)
- Increase batch size if memory allows

## Next Steps

- Increase epochs (2-3) for better performance
- Tune LoRA rank (r) and learning rate
- Try different base models (Qwen2.5, Mistral)
- Full test set evaluation (set `FULL_TEST_EVAL=True`)
- Experiment with different prompt templates

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- Week 4 notebook for comparison with traditional ML approaches

---

**Note**: This notebook is designed to run end-to-end in Google Colab. For local execution, ensure you have a GPU and adjust paths accordingly.

