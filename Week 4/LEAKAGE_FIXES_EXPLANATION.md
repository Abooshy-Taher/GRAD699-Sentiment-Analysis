# Data Leakage Fixes and Pipeline Explanation

## Summary

This document explains the key data leakage fixes implemented in the Week 4 notebook and how the chronological validation works.

## Key Data Leakage Fixes

### 1. **Chronological Splitting (Instead of Random Split)**

**Previous Issue:**
- Used `train_test_split` with `random_state` and `stratify`
- This randomly shuffled data, causing future information to leak into training

**Fix:**
- Sort data by timestamp first
- Split chronologically: 70% train (oldest), 15% validation (middle), 15% test (most recent)
- Ensures no temporal overlap between splits
- Most realistic for time-series prediction tasks

```python
# Chronological splits
df_train = df.iloc[:n_train].copy()
df_val = df.iloc[n_train:n_train + n_val].copy()
df_test = df.iloc[n_train + n_val:].copy()
```

### 2. **Target Definition: Rating-Based (Not Text-Derived)**

**Previous Issue:**
- Created target using VADER sentiment scores from text
- Then trained models to predict sentiment from the same text
- This is circular: "predicting a label created from the same text with a similar method"

**Fix:**
- Use **rating** (1-5 stars) as ground truth label
- Define: negative = rating ≤ 2, positive = rating ≥ 4
- Rating is the actual user-provided label, independent of text analysis
- No circular dependency

```python
# Define negative reviews (rating <= 2) and positive (rating >= 4)
df['is_negative'] = (df['rating'] <= 2).astype(int)
```

### 3. **Feature Engineering: Fit on Train Only**

**Previous Issue:**
- VADER sentiment scores computed on full dataset before splitting
- Any statistics computed on full dataset leak test information

**Fix:**
- All transforms (TF-IDF, StandardScaler) fit on training data only
- Use scikit-learn Pipeline where possible
- Transform validation and test sets using train-fitted transformers

```python
# TF-IDF fit on train, transform val/test
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train_text)  # Fit on train
X_val_tfidf = tfidf.transform(X_val_text)           # Transform val (no fit)
X_test_tfidf = tfidf.transform(X_test_text)         # Transform test (no fit)
```

### 4. **EDA: Descriptive Only (No Threshold Learning)**

**Previous Issue:**
- EDA might have been used to set thresholds used in modeling
- Learning thresholds from full data leaks information

**Fix:**
- EDA performed on train set only (or train+val for more data)
- EDA is strictly descriptive
- No thresholds or decision rules learned from EDA
- EDA results don't influence model features or decisions

### 5. **Time-Aware Cross-Validation**

**Previous Issue:**
- Standard K-fold CV doesn't respect temporal ordering
- Can cause temporal leakage in cross-validation

**Fix:**
- Use `TimeSeriesSplit` for cross-validation
- Ensures training folds always come before validation folds in time
- Properly evaluates model stability over time
- Used for model assessment and hyperparameter tuning

```python
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, val_idx in tscv.split(df_train_val):
    # train_idx always contains earlier timepoints than val_idx
    ...
```

## Chronological Validation Workflow

### Step 1: Load and Clean Data
- Load data
- Basic cleaning (remove nulls, convert types)
- Sort by timestamp (critical!)

### Step 2: Define Target (Before Splitting)
- Create target from rating (ground truth)
- Target is independent of text features

### Step 3: Chronological Split
- 70% train (oldest data)
- 15% validation (middle)
- 15% test (most recent)
- Sanity checks: verify no temporal overlap

### Step 4: Feature Engineering
- Create time features (no fitting needed)
- Fit text transforms (TF-IDF) on train only
- Transform val/test using train-fitted transformers

### Step 5: Model Training
- Train models on training set
- All feature engineering fit on train
- Evaluate on validation set

### Step 6: Model Selection
- Compare models using validation performance
- Use time-aware CV for stability assessment
- Select best model based on validation metrics

### Step 7: Final Test Evaluation
- **Test set used ONLY ONCE**
- Evaluate selected model on test set
- Report final unbiased performance metrics

## Pipeline Structure

The notebook follows this structure (A-M sections):

- **A. Imports & Config**: Libraries and configuration
- **B. Load Data**: Load dataset
- **C. Data Cleaning / Preprocessing**: Basic cleaning (no statistics)
- **D. Define Target**: Rating-based labels (not text-derived)
- **E. Chronological Split**: Time-based train/val/test split
- **F. Feature Engineering**: Fit on train only
- **G. EDA**: Descriptive analysis (train set only)
- **H. Baselines**: Simple baselines
- **I. Models**: Multiple models (text-only, time-only, combined)
- **J. Validation & Metrics**: Time-aware CV
- **K. Final Test Evaluation**: One-time test set evaluation
- **L. Error Analysis**: Analyze model errors
- **M. Conclusions**: Summary and next steps

## Why These Fixes Matter

1. **Realistic Performance Estimates**: No leakage means test performance reflects real-world performance
2. **Defensible Methodology**: Can explain to professor why approach is correct
3. **Temporal Validity**: Chronological splits respect time ordering, critical for time-series tasks
4. **No Circular Logic**: Using rating (not text-derived sentiment) as target avoids circular dependencies
5. **Proper Validation**: Time-aware CV properly assesses model stability over time

## Models Implemented

1. **Baseline 1**: Majority class classifier
2. **Baseline 2**: Simple time-based heuristic (learned from train only)
3. **Model 1**: TF-IDF + Logistic Regression (text-only)
4. **Model 2**: TF-IDF + Linear SVM (text-only)
5. **Model 3**: Logistic Regression on time features (time-only)
6. **Model 4**: TF-IDF + time features combined (text+time)
7. **Model 5**: Sentence-transformers embeddings + Logistic Regression (language model)

All models follow the fit-on-train-only principle.

