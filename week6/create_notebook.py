#!/usr/bin/env python3
"""Script to generate the complete Week6 notebook with all sections."""

import json

notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.9.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_markdown(source):
    """Add markdown cell."""
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': source.split('\n')
    })

def add_code(source):
    """Add code cell."""
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source.split('\n')
    })

# Title
add_markdown("""# Week 6: Final Sentiment Analysis Pipeline

## End-to-End Thesis-Quality Experiment Pipeline

This notebook combines all components into a clean, reproducible, Colab-ready pipeline:
- Data loading & cleaning
- Chronological splitting (prevents temporal leakage)
- Feature engineering (TF-IDF + time features, fit on train only)
- Baseline models (TF-IDF + Logistic Regression)
- Transformer fine-tuning (DistilBERT)
- Unsloth fine-tuning (Llama-3.1-8B with LoRA)
- Final evaluation & comparison

**Reproducibility**: seed=319302 throughout
**Data Leakage Prevention**: Chronological splits, transforms fit on train only
**Colab-Ready**: Auto-detects data paths, includes GPU checks""")

# Section A
add_markdown("## A. Environment Setup (Colab installs, GPU check)")

add_code("""# Install required packages for Colab
# Skip if running locally and packages are already installed
import sys
import subprocess

def install_if_missing(package):
    try:
        __import__(package.split("==")[0].split(">=")[0].split("[")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Core ML packages
install_if_missing("pandas>=2.0.0")
install_if_missing("numpy>=1.24.0")
install_if_missing("scikit-learn>=1.3.0")
install_if_missing("matplotlib>=3.7.0")
install_if_missing("seaborn>=0.12.0")

# Transformer packages
install_if_missing("transformers>=4.40.0")
install_if_missing("datasets>=2.18.0")
install_if_missing("evaluate>=0.4.1")
install_if_missing("accelerate>=0.20.0")
install_if_missing("torch>=2.0.0")

print("✓ Core packages installed")""")

add_code("""# Optional: Mount Google Drive (uncomment if needed)
# from google.colab import drive
# drive.mount('/content/drive')
# print("✓ Google Drive mounted")""")

add_code("""# Check GPU availability
import torch

if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    DEVICE = "cuda"
else:
    print("⚠️  CPU mode (GPU recommended for transformer/unsloth training)")
    DEVICE = "cpu"
""")

# Section B
add_markdown("## B. Imports & Global Config (seed=319302, paths, flags)")

add_code("""import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from scipy.sparse import hstack

# Transformer imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import evaluate

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
RANDOM_STATE = 319302
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# FAST RUN toggles
SAMPLE_FRAC = 1.0  # Use 1.0 for full dataset, < 1.0 for quick tests
MAX_ROWS = None  # None for all rows, or set limit for debug
EPOCHS_BERT = 1  # Epochs for DistilBERT fine-tuning
EPOCHS_UNSLOTH = 1  # Epochs for Unsloth fine-tuning

# Output directories
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("✓ Imports complete")
print(f"✓ Random seed: {RANDOM_STATE}")
print(f"✓ Sample fraction: {SAMPLE_FRAC}")""")

# Section C
add_markdown("## C. Load Data (auto-detect paths, show shape/columns)")

add_code("""def load_dataset():
    \"\"\"
    Load Amazon_Data.csv from multiple possible locations.
    Auto-detects Colab paths and local paths.
    \"\"\"
    possible_paths = [
        # Colab paths
        "/content/drive/MyDrive/Amazon_Data.csv",
        "/content/Amazon_Data.csv",
        # Local paths
        "../Amazon_Data.csv",
        "Amazon_Data.csv",
        os.path.join(os.path.expanduser("~"), "Desktop/HU Classes/GRAD699/Sentiment Analysis/Amazon_Data.csv")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Found file at: {path}")
            print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            return df
    
    raise FileNotFoundError(
        "Could not find Amazon_Data.csv in any expected location.\\n"
        "Please place Amazon_Data.csv in one of:\\n"
        "  - /content/drive/MyDrive/Amazon_Data.csv (Google Drive)\\n"
        "  - /content/Amazon_Data.csv (Colab upload)\\n"
        "  - ./Amazon_Data.csv (local)"
    )

df = load_dataset()

# Apply sampling if needed
if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"✓ Sampled to {len(df):,} rows ({SAMPLE_FRAC*100:.1f}%)")

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)
    print(f"✓ Limited to {len(df):,} rows (MAX_ROWS={MAX_ROWS})")""")

# Section D
add_markdown("## D. Cleaning & Basic Preprocessing (text/timestamp/rating, dedupe if needed)")

add_code("""def clean_data(df):
    \"\"\"Clean dataset: remove nulls, convert timestamp, remove empty text.\"\"\"
    # Keep only necessary columns
    df = df[['text', 'rating', 'timestamp']].copy()
    
    print(f"Before cleaning: {len(df):,} rows")
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"After removing nulls: {len(df):,} rows")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    print(f"After timestamp conversion: {len(df):,} rows")
    
    # Remove empty text reviews
    df = df[df['text'].astype(str).str.len() > 0].copy()
    print(f"After removing empty text: {len(df):,} rows")
    
    # Sort by timestamp (critical for chronological splitting)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\\nRating distribution:")
    print(df['rating'].value_counts().sort_index())
    
    return df

df = clean_data(df)""")

# Section E
add_markdown("## E. Target Definition (ternary + optional binary helper)")

add_code("""def create_sentiment_labels(df):
    \"\"\"
    Create sentiment labels from ratings:
    - Rating ≤ 2 → Negative (0)
    - Rating = 3 → Neutral (1)
    - Rating ≥ 4 → Positive (2)
    \"\"\"
    df = df.copy()
    
    def rating_to_label(rating):
        if rating <= 2:
            return 0  # Negative
        elif rating == 3:
            return 1  # Neutral
        else:
            return 2  # Positive
    
    df['sentiment_label'] = df['rating'].apply(rating_to_label)
    df['is_negative'] = (df['rating'] <= 2).astype(int)  # Binary helper
    
    label_counts = df['sentiment_label'].value_counts().sort_index()
    print("=" * 60)
    print("TARGET DISTRIBUTION (Rating-Based Labels)")
    print("=" * 60)
    print(f"  Negative (1-2 stars): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Neutral (3 stars):     {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Positive (4-5 stars):  {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/len(df)*100:.1f}%)")
    print("=" * 60)
    
    return df

df = create_sentiment_labels(df)
y = df['sentiment_label'].values  # Ternary labels (0=negative, 1=neutral, 2=positive)""")

# Section F
add_markdown("## F. Chronological Split (70/15/15 train/val/test with date ranges printed)")

add_code("""def chronological_split(df, y, train_ratio=0.70, val_ratio=0.15):
    \"\"\"
    Split dataframe chronologically by timestamp.
    Returns: df_train, df_val, df_test, y_train, y_val, y_test
    \"\"\"
    n_total = len(df)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Chronological splits (data already sorted by timestamp)
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()
    
    # Extract targets
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]
    
    print("=" * 60)
    print("CHRONOLOGICAL SPLIT COMPLETE")
    print("=" * 60)
    print(f"Train set: {len(df_train):,} samples ({len(df_train)/n_total*100:.1f}%)")
    print(f"  Date range: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"\\nValidation set: {len(df_val):,} samples ({len(df_val)/n_total*100:.1f}%)")
    print(f"  Date range: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"\\nTest set: {len(df_test):,} samples ({len(df_test)/n_total*100:.1f}%)")
    print(f"  Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    
    # Assert no temporal overlap
    assert df_train['timestamp'].max() <= df_val['timestamp'].min(), "Train/Val overlap!"
    assert df_val['timestamp'].max() <= df_test['timestamp'].min(), "Val/Test overlap!"
    print("\\n✓ Sanity checks passed: no temporal overlap")
    print("=" * 60)
    
    return df_train, df_val, df_test, y_train, y_val, y_test

df_train, df_val, df_test, y_train, y_val, y_test = chronological_split(df, y)""")

# Section G
add_markdown("## G. Feature Engineering (time-of-day features + TF-IDF pipeline; fit on train only)")

add_code("""def create_time_features(df):
    \"\"\"Create time-based features (circular encoding for hour).\"\"\"
    df = df.copy()
    df['review_hour'] = df['timestamp'].dt.hour
    df['review_day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Circular encoding for hour (preserves 23-0 proximity)
    df['hour_sin'] = np.sin(2 * np.pi * df['review_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['review_hour'] / 24)
    df['is_weekend'] = (df['review_day_of_week'] >= 5).astype(int)
    
    return df

# Apply to all splits (no fitting needed)
df_train = create_time_features(df_train)
df_val = create_time_features(df_val)
df_test = create_time_features(df_test)

# Time features for modeling
TIME_FEATURES = ['hour_sin', 'hour_cos', 'is_weekend']

print("✓ Time features created on all splits")
print(f"  Features: {TIME_FEATURES}")""")

# Section H
add_markdown("## H. EDA (strictly descriptive: label distribution by hour/day/month; no leakage)")

add_code("""# Descriptive EDA on TRAIN set only (no leakage)
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS (Train Set Only)")
print("=" * 60)

# Distribution by hour
sentiment_by_hour = df_train.groupby('review_hour').agg({
    'sentiment_label': lambda x: (x == 0).mean(),  # Negative rate
}).reset_index()
sentiment_by_hour.columns = ['hour', 'negative_rate']
sentiment_by_hour['n_reviews'] = df_train.groupby('review_hour').size().values
sentiment_by_hour['positive_rate'] = df_train.groupby('review_hour')['sentiment_label'].apply(lambda x: (x == 2).mean()).values

print("\\nSentiment by Hour (Train Set Only):")
print(sentiment_by_hour[['hour', 'n_reviews', 'negative_rate', 'positive_rate']].head(10))

# Optional: Plot (skip if in headless mode)
try:
    plt.figure(figsize=(12, 5))
    plt.plot(sentiment_by_hour['hour'], sentiment_by_hour['negative_rate'], marker='o', linewidth=2)
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Negative Review Rate')
    plt.title('Negative Review Rate by Hour (Train Set Only)')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'negative_rate_by_hour.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\\n✓ Plot saved to outputs/figures/negative_rate_by_hour.png")
except Exception as e:
    print(f"\\n⚠️  Skipping plot: {e}")

print("\\n✓ EDA completed (descriptive only, no thresholds learned)")""")

# Section I
add_markdown("## I. Baseline Models")

add_code("""def run_tfidf_baseline(df_train, df_val, df_test, y_train, y_val, y_test):
    \"\"\"
    Train TF-IDF + Logistic Regression baseline.
    Returns: model dict with tfidf, scaler, clf, time_features
    \"\"\"
    print("Training TF-IDF + Logistic Regression baseline...")
    
    # Fit TF-IDF on train only
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(df_train['text'].values)
    X_val_tfidf = tfidf.transform(df_val['text'].values)
    X_test_tfidf = tfidf.transform(df_test['text'].values)
    
    # Scale time features (fit on train only)
    scaler = StandardScaler()
    X_train_time = scaler.fit_transform(df_train[TIME_FEATURES].values)
    X_val_time = scaler.transform(df_val[TIME_FEATURES].values)
    X_test_time = scaler.transform(df_test[TIME_FEATURES].values)
    
    # Combine features
    X_train_combined = hstack([X_train_tfidf, X_train_time])
    X_val_combined = hstack([X_val_tfidf, X_val_time])
    X_test_combined = hstack([X_test_tfidf, X_test_time])
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    clf.fit(X_train_combined, y_train)
    
    # Predictions
    y_val_pred = clf.predict(X_val_combined)
    y_test_pred = clf.predict(X_test_combined)
    
    # Metrics
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_acc = accuracy_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"  Validation F1 (macro): {val_f1:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    return {
        'tfidf': tfidf,
        'scaler': scaler,
        'clf': clf,
        'time_features': TIME_FEATURES,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred
    }

baseline_model = run_tfidf_baseline(df_train, df_val, df_test, y_train, y_val, y_test)
print("\\n✓ Baseline model complete")""")

# Section J - DistilBERT
add_markdown("## J. Transformer Fine-tuning (standard HF classifier)")

add_code("""def run_distilbert_classifier(df_train, df_val, df_test, y_train, y_val, y_test):
    \"\"\"
    Fine-tune DistilBERT for ternary sentiment classification.
    \"\"\"
    print("Fine-tuning DistilBERT for sentiment classification...")
    
    model_name = "distilbert-base-uncased"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    ).to(DEVICE)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    train_dataset = Dataset.from_dict({'text': df_train['text'].tolist(), 'label': y_train.tolist()})
    val_dataset = Dataset.from_dict({'text': df_val['text'].tolist(), 'label': y_val.tolist()})
    test_dataset = Dataset.from_dict({'text': df_test['text'].tolist(), 'label': y_test.tolist()})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Compute metrics
    metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'f1': metric.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'accuracy': accuracy_score(labels, predictions)
        }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "distilbert_sentiment"),
        num_train_epochs=EPOCHS_BERT,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    # Get predictions
    val_predictions = trainer.predict(val_dataset)
    test_predictions = trainer.predict(test_dataset)
    y_val_pred = np.argmax(val_predictions.predictions, axis=1)
    y_test_pred = np.argmax(test_predictions.predictions, axis=1)
    
    print(f"  Validation F1 (macro): {val_results['eval_f1']:.4f}")
    print(f"  Validation Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_results['eval_f1']:.4f}")
    print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    # Save model
    model.save_pretrained(os.path.join(MODELS_DIR, "distilbert_sentiment"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "distilbert_sentiment"))
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'val_f1': val_results['eval_f1'],
        'val_acc': val_results['eval_accuracy'],
        'test_f1': test_results['eval_f1'],
        'test_acc': test_results['eval_accuracy'],
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred
    }

# Uncomment to run DistilBERT fine-tuning (may take time)
# bert_model = run_distilbert_classifier(df_train, df_val, df_test, y_train, y_val, y_test)
# print("\\n✓ DistilBERT fine-tuning complete")
print("⚠️  DistilBERT fine-tuning skipped (uncomment to run)")
bert_model = None  # Placeholder""")

# Section K - Unsloth
add_markdown("## K. Unsloth Fine-tuning (from week5_unsloth_sentiment.ipynb)")

add_code("""# Install Unsloth (Colab-ready)
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    print("✓ Unsloth already installed")
except ImportError:
    print("Installing Unsloth...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[colab-new]", "-q", "--no-deps"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "xformers<0.0.27", "trl<0.9.0", "peft<0.10.0", "bitsandbytes<0.43.0", "-q"])
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    print("✓ Unsloth installed")""")

add_code("""def label_to_string(label):
    \"\"\"Convert numeric label to string.\"\"\"
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[label]

def string_to_label(label_str):
    \"\"\"Convert string label to numeric.\"\"\"
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    label_str_lower = label_str.strip().lower()
    for key, value in label_map.items():
        if key.lower() in label_str_lower or label_str_lower in key.lower():
            return value
    return 1  # Default to Neutral

def create_instruction_prompt(text):
    \"\"\"Create instruction prompt for sentiment classification.\"\"\"
    return f\"\"\"Classify the sentiment of this review as one of: Negative, Neutral, Positive.

Review: {text}

Answer:\"\"\"

def prepare_unsloth_datasets(df_split, y_split):
    \"\"\"Prepare dataset in instruction format for Unsloth.\"\"\"
    texts = []
    labels = []
    for idx, row in df_split.iterrows():
        instruction = create_instruction_prompt(row['text'])
        texts.append(instruction)
        labels.append(label_to_string(y_split[idx]))
    
    return Dataset.from_dict({'text': texts, 'label': labels})

print("✓ Unsloth helper functions defined")""")

# Continue with full Unsloth function - need to split into multiple cells due to length
add_code("""def run_unsloth_finetune(df_train, df_val, df_test, y_train, y_val, y_test):
    \"\"\"Fine-tune Llama-3.1-8B with Unsloth for sentiment classification.\"\"\"
    print("Fine-tuning Llama-3.1-8B with Unsloth...")
    
    # Load model
    model_name = "unsloth/llama-3.1-8b-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=RANDOM_STATE,
    )
    
    # Prepare datasets
    train_dataset = prepare_unsloth_datasets(df_train, y_train)
    val_dataset = prepare_unsloth_datasets(df_val, y_val)
    test_dataset = prepare_unsloth_datasets(df_test, y_test)
    
    # Format for training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    def format_dataset(examples):
        inputs = examples['text']
        outputs = examples['label']
        texts = [f"{inp}{out}" for inp, out in zip(inputs, outputs)]
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding=False)
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    train_dataset_formatted = train_dataset.map(format_dataset, batched=True, remove_columns=train_dataset.column_names)
    val_dataset_formatted = val_dataset.map(format_dataset, batched=True, remove_columns=val_dataset.column_names)
    test_dataset_formatted = test_dataset.map(format_dataset, batched=True, remove_columns=test_dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=EPOCHS_UNSLOTH,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        output_dir=os.path.join(MODELS_DIR, "unsloth_sentiment_model"),
        optim="adamw_8bit",
        load_best_model_at_end=True,
        report_to="none",
        seed=RANDOM_STATE,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset_formatted,
        eval_dataset=val_dataset_formatted,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )
    
    # Train
    trainer_stats = trainer.train()
    print(f"  Training loss: {trainer_stats.training_loss:.4f}")
    
    # Save model
    model.save_pretrained(os.path.join(MODELS_DIR, "unsloth_sentiment_model"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "unsloth_sentiment_model"))
    
    # Inference function
    FastLanguageModel.for_inference(model)
    
    def predict_sentiment(text):
        prompt = create_instruction_prompt(text)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        answer_words = answer.split()
        if len(answer_words) > 0:
            predicted_label = answer_words[0].strip()
        else:
            predicted_label = answer.strip()
        return label_to_string(string_to_label(predicted_label))
    
    # Evaluate (sample for speed)
    sample_size = min(1000, len(df_test))
    test_sample_idx = np.random.choice(len(df_test), sample_size, replace=False)
    test_texts_sample = df_test.iloc[test_sample_idx]['text'].values
    y_test_true_sample = y_test[test_sample_idx]
    
    print(f"  Evaluating on {sample_size} test samples...")
    y_test_pred_sample = []
    for text in test_texts_sample:
        pred = predict_sentiment(text)
        y_test_pred_sample.append(pred)
    
    y_test_pred_numeric = np.array([string_to_label(pred) for pred in y_test_pred_sample])
    y_test_true_numeric = np.array([string_to_label(label_to_string(label)) for label in y_test_true_sample])
    
    test_f1 = f1_score(y_test_true_numeric, y_test_pred_numeric, average='macro')
    test_acc = accuracy_score(y_test_true_numeric, y_test_pred_numeric)
    
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'predict_sentiment': predict_sentiment,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'y_test_pred_numeric': y_test_pred_numeric,
        'y_test_true_numeric': y_test_true_numeric
    }

# Uncomment to run Unsloth fine-tuning (requires GPU, may take time)
# unsloth_model = run_unsloth_finetune(df_train, df_val, df_test, y_train, y_val, y_test)
# print("\\n✓ Unsloth fine-tuning complete")
print("⚠️  Unsloth fine-tuning skipped (uncomment to run)")
unsloth_model = None  # Placeholder""")

# Section L
add_markdown("## L. Final Comparison & Export")

add_code("""def evaluate_and_save(baseline_model, bert_model, unsloth_model, y_test, output_dir):
    \"\"\"Create comparison table and save results.\"\"\"
    results = []
    
    # Baseline
    results.append({
        'Model': 'TF-IDF + Logistic Regression',
        'Test F1 (macro)': baseline_model['test_f1'],
        'Test Accuracy': baseline_model['test_acc']
    })
    
    # DistilBERT
    if bert_model is not None:
        results.append({
            'Model': 'DistilBERT (Fine-tuned)',
            'Test F1 (macro)': bert_model['test_f1'],
            'Test Accuracy': bert_model['test_acc']
        })
    
    # Unsloth
    if unsloth_model is not None:
        results.append({
            'Model': 'Unsloth (Llama-3.1-8B)',
            'Test F1 (macro)': unsloth_model['test_f1'],
            'Test Accuracy': unsloth_model['test_acc']
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    print("=" * 60)
    print("FINAL MODEL COMPARISON (Test Set)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"\\n✓ Results saved to {output_dir}/model_comparison.csv")
    
    # Confusion matrices
    if baseline_model is not None:
        cm_baseline = confusion_matrix(y_test, baseline_model['y_test_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - TF-IDF Baseline')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'confusion_matrix_baseline.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if bert_model is not None:
        cm_bert = confusion_matrix(y_test, bert_model['y_test_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - DistilBERT')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'confusion_matrix_distilbert.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if unsloth_model is not None:
        cm_unsloth = confusion_matrix(unsloth_model['y_test_true_numeric'], unsloth_model['y_test_pred_numeric'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_unsloth, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Unsloth')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', 'confusion_matrix_unsloth.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\\n✓ Confusion matrices saved to {output_dir}/figures/")
    
    return results_df

results_df = evaluate_and_save(baseline_model, bert_model, unsloth_model, y_test, OUTPUT_DIR)""")

# Final summary
add_markdown("""---
## Summary

**Pipeline Complete** ✓

- Data loaded and cleaned
- Chronological split (70/15/15)
- Baseline model trained (TF-IDF + Logistic Regression)
- Transformer fine-tuning available (DistilBERT)
- Unsloth fine-tuning available (Llama-3.1-8B)
- Results saved to `outputs/`

**Note**: To run DistilBERT or Unsloth fine-tuning, uncomment the respective sections above.

**Reproducibility**: All results use seed=319302

**Data Leakage Prevention**: ✓ Chronological splits, transforms fit on train only""")

# Save notebook
with open('week6/Week6_Final_Sentiment_Pipeline.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook created successfully!")
