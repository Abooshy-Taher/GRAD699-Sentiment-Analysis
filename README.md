# Sentiment Analysis Project: Time-of-Day Effects on Review Sentiment

A comprehensive sentiment analysis project analyzing Amazon product reviews to study how time-of-day affects review sentiment. This project implements a rigorous machine learning pipeline with strict data leakage prevention protocols.

## 📋 Project Overview

This project performs sentiment analysis on Amazon product reviews with a focus on:
- **Temporal patterns**: How time of day affects whether customers leave positive, negative, or neutral reviews
- **Data leakage prevention**: Strict chronological splitting and proper feature engineering
- **Multiple model comparison**: TF-IDF, time features, embeddings, and BERT models
- **Ternary classification**: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)

## 🎯 Objectives

- Analyze sentiment patterns across different hours of the day
- Build and compare multiple ML models (Text-only, Time-only, Combined, Word2Vec, BERT)
- Predict review sentiment using **rating-based ground truth labels** (not text-derived)
- Prevent data leakage through chronological splitting and proper feature engineering
- Evaluate models using time-aware cross-validation
- Identify optimal time windows for feedback requests

## 📁 Project Structure

```
Sentiment Analysis/
├── Amazon_Data.csv                      # Dataset (Amazon product reviews)
├── Week 1/                              # Initial experiments and feasibility study
│   ├── Feasibility Study Summary.docx
│   └── Sentiment Analysis Experiment.R
├── Week 2/                              # Comprehensive testing framework
│   ├── ANALYSIS_CONCLUSION.md
│   ├── VSCodeExperiment.ipynb
│   └── VSCodeExperiment_Organized.ipynb
├── Week 3/                              # Organized code with comments
│   └── Organized Code w Comments.ipynb
├── Week 4/                              # ⭐ Main pipeline (production-ready)
│   ├── Sentiment_Analysis_Pipeline.ipynb    # Complete ML pipeline with leakage fixes
│   └── LEAKAGE_FIXES_EXPLANATION.md         # Detailed explanation of fixes
├── week5/                                # 🚀 Unsloth LLM fine-tuning
│   ├── week5_unsloth_sentiment.ipynb        # Unsloth fine-tuning notebook (Colab-ready)
│   ├── week5_utils.py                       # Utility functions
│   └── README.md                            # Week 5 documentation
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Installation below)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Sentiment Analysis"
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy gensim transformers torch
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

**Note**: For BERT models, you may also need:
- PyTorch (installed via transformers dependency)
- Hugging Face transformers library

## 📊 Dataset

The dataset contains Amazon product reviews with the following columns:
- `rating`: Star rating (1-5) - **used as ground truth for sentiment labels**
- `text`: Review text (main feature)
- `timestamp`: Review timestamp (used for chronological splitting and time features)
- Additional metadata fields

**Important Notes**: 
- The dataset file (`Amazon_Data.csv`) is not included in this repository due to its large size (>200MB)
- You will need to provide your own dataset or download it separately
- Ensure the CSV file is placed in the root directory with the name `Amazon_Data.csv`
- The dataset should have at minimum: `text`, `rating`, and `timestamp` columns

## 🔬 Main Pipeline: Week 4 Notebook

The **Week 4 notebook** (`Sentiment_Analysis_Pipeline.ipynb`) contains the production-ready pipeline with strict data leakage prevention:

### Pipeline Structure (Sections A-M):

1. **A. Imports & Config**: Libraries and configuration
2. **B. Load Data**: Load and inspect dataset
3. **C. Data Cleaning / Preprocessing**: Basic cleaning (before splitting)
4. **D. Define Target**: Rating-based ternary labels (Negative/Neutral/Positive)
5. **E. Chronological Split**: Time-based train/val/test split (70/15/15)
6. **F. Feature Engineering**: Fit transformers on train only
7. **G. EDA**: Exploratory analysis on train set only
8. **H. Baselines**: Majority class + time-based heuristic
9. **I. Models**: Multiple model implementations
10. **J. Validation & Metrics**: Time-aware cross-validation
11. **K. Final Test Evaluation**: One-time test set evaluation
12. **L. Error Analysis**: Misclassification analysis
13. **M. Conclusions**: Summary and next steps

### Models Implemented:

1. **Baseline 1**: Majority class classifier
2. **Baseline 2**: Time-based heuristic (most common sentiment per hour)
3. **Model 1**: TF-IDF + Logistic Regression (text-only)
4. **Model 2**: Time features only (Logistic Regression on hour/day/weekend)
5. **Model 3**: Text+Time Combined (TF-IDF + time features)
6. **Model 4**: Word2Vec embeddings (averaged) + Logistic Regression
7. **Model 5**: BERT (DistilBERT) for sentiment classification

### Key Features:

- ✅ **Ternary Classification**: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)
- ✅ **Rating-Based Labels**: Uses star ratings as ground truth (not text-derived sentiment)
- ✅ **Chronological Splitting**: 70% train (oldest), 15% validation, 15% test (most recent)
- ✅ **No Data Leakage**: All transforms fit on training data only
- ✅ **Time-Aware Validation**: Uses TimeSeriesSplit for cross-validation
- ✅ **Comprehensive Metrics**: F1 (macro), ROC-AUC, Precision, Recall for multi-class
- ✅ **Error Analysis**: Confusion matrix, misclassification examples, performance by hour

## 🛡️ Data Leakage Prevention

The Week 4 notebook implements strict protocols to prevent data leakage:

1. **Chronological Splitting**: Data split by timestamp (not random), ensuring temporal ordering
2. **Rating-Based Targets**: Labels derived from star ratings (ground truth), not from text analysis
3. **Fit-on-Train-Only**: All feature engineering (TF-IDF, scalers) fit on training data only
4. **Time-Aware CV**: Uses TimeSeriesSplit to respect temporal ordering in cross-validation
5. **Test Set Isolation**: Test set used ONLY once for final evaluation

For detailed explanation, see `Week 4/LEAKAGE_FIXES_EXPLANATION.md`.

## 📈 Usage

### Running the Main Pipeline (Week 4):

1. Open the Jupyter notebook:
```bash
jupyter notebook "Week 4/Sentiment_Analysis_Pipeline.ipynb"
```

2. Run cells sequentially:
   - Sections A-E: Data loading, cleaning, target definition, and splitting
   - Section F: Feature engineering
   - Section G: Exploratory data analysis
   - Section H: Baseline models
   - Section I: All ML models (including BERT)
   - Section J: Time-aware cross-validation
   - Section K: Final test evaluation
   - Sections L-M: Error analysis and conclusions

### Running Previous Notebooks:

- **Week 2**: `Week 2/VSCodeExperiment.ipynb` - Initial comprehensive testing framework
- **Week 3**: `Week 3/Organized Code w Comments.ipynb` - Organized code with detailed comments

## 📝 Key Methodological Improvements

### Target Definition:
- **Ternary Classification**: 
  - Negative: 1-2 stars (label = 0)
  - Neutral: 3 stars (label = 1)
  - Positive: 4-5 stars (label = 2)
- **No Circular Logic**: Labels come from ratings (user-provided), not from analyzing the text

### Splitting Strategy:
- **Chronological**: Most recent 15% = test set (most realistic evaluation)
- **No Temporal Overlap**: Ensures train < validation < test in time
- **Time-Aware CV**: Cross-validation respects temporal ordering

### Feature Engineering:
- **Text Features**: TF-IDF (fit on train, transform val/test)
- **Time Features**: Hour (circular encoding: sin/cos), weekend indicator
- **Embeddings**: Word2Vec (trained on train only), BERT (pre-trained)

### Evaluation:
- **Multi-Class Metrics**: F1 (macro), ROC-AUC (one-vs-rest), Precision/Recall (macro)
- **Time-Aware CV**: 3-fold TimeSeriesSplit for model stability assessment
- **Single Test Use**: Test set evaluated only once at the end

## 🛠️ Technologies Used

### Core Libraries:
- **Python**: Data processing and analysis
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models, pipelines, metrics
- **Matplotlib/Seaborn**: Visualizations

### ML & NLP Libraries:
- **Scikit-learn**: Logistic Regression, TF-IDF, StandardScaler, TimeSeriesSplit
- **Gensim**: Word2Vec embeddings
- **Transformers**: BERT/DistilBERT models (Hugging Face)
- **PyTorch**: Backend for transformers

### Analysis Tools:
- **SciPy**: Statistical tests (for EDA)
- **Time-series Tools**: TimeSeriesSplit for temporal validation

## 📊 Expected Results

The pipeline produces:
- Model comparison table (F1, ROC-AUC, Precision, Recall)
- Best model selection based on validation performance
- Final test set evaluation (unbiased performance estimate)
- Confusion matrix for ternary classification
- Error analysis showing misclassification patterns
- Performance analysis by hour of day
- Time-aware cross-validation scores

## 🔍 Previous Notebooks

### Week 2: Comprehensive Testing Framework
- Initial testing framework with 10 different tests
- Comparison of multiple ML models
- Statistical analysis

### Week 3: Organized Code
- Well-commented code structure
- Initial attempt at preventing data leakage
- Business insights and recommendations

### Week 4: Production Pipeline ⭐
- **Recommended for final analysis**
- Strict data leakage prevention
- Chronological splitting
- Rating-based targets
- Complete model comparison
- Production-ready code

### Week 5: Unsloth LLM Fine-tuning 🚀
- **LLM fine-tuning with Unsloth**
- Fine-tunes Llama-3.1-8B using LoRA + 4-bit quantization
- Instruction-following format for sentiment classification
- Same chronological split and leakage prevention as Week 4
- Colab-ready notebook with GPU support
- See [week5/README.md](week5/README.md) for details

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

This project is for academic purposes.

## 👤 Author

Abdullah Ahmad Adel Al-Taher
- Course: GRAD699
- Institution: Harrisburg University of Science and Technology
- Project: Sentiment Analysis - Time-of-Day Effects on Review Sentiment

## 🙏 Acknowledgments

- Amazon for providing review data
- Open-source community for excellent libraries (scikit-learn, Hugging Face, Gensim)
- Hugging Face for pre-trained BERT models

## 📚 References

- Week 4 notebook: `Week 4/Sentiment_Analysis_Pipeline.ipynb`
- Data leakage explanation: `Week 4/LEAKAGE_FIXES_EXPLANATION.md`
- Week 5 notebook: `week5/week5_unsloth_sentiment.ipynb`
- Week 5 documentation: `week5/README.md`

---

**Note**: 
- The Week 4 notebook is the recommended starting point for understanding the complete pipeline with proper data leakage prevention protocols.
- Week 5 introduces LLM fine-tuning with Unsloth for advanced sentiment classification.
