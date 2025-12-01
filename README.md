# Sentiment Analysis Project

A comprehensive sentiment analysis project analyzing Amazon product reviews using multiple machine learning techniques and statistical methods.

## 📋 Project Overview

This project performs sentiment analysis on Amazon product reviews, comparing different sentiment analysis methods (VADER and TextBlob) and evaluating multiple machine learning models to predict review sentiment.

## 🎯 Objectives

- Compare sentiment analysis methods (VADER vs TextBlob)
- Evaluate multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Analyze sentiment patterns across ratings and time periods
- Perform statistical tests to identify significant relationships
- Feature engineering and importance analysis

## 📁 Project Structure

```
Sentiment Analysis/
├── Amazon_Data.csv          # Dataset (Amazon product reviews)
├── Week 1/                   # Initial experiments and feasibility study
│   ├── Feasibility Study Summary.docx
│   └── Sentiment Analysis Experiment.R
├── Week 2/                   # Comprehensive testing framework
│   └── VSCodeExperiment.ipynb
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Sentiment Analysis"
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy vaderSentiment textblob
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The dataset contains Amazon product reviews with the following columns:
- `rating`: Star rating (1-5)
- `title`: Review title
- `text`: Review text
- `timestamp`: Review timestamp
- Additional metadata fields

**Note**: 
- The dataset file (`Amazon_Data.csv`) is not included in this repository due to its large size (>200MB)
- You will need to provide your own dataset or download it separately
- Ensure the CSV file is placed in the root directory with the name `Amazon_Data.csv`
- The dataset should have at minimum: `text`, `rating`, and `timestamp` columns

## 🔬 Testing Framework

The project includes a comprehensive testing framework with 10 different tests:

### Test 1: Sentiment Analysis Methods Comparison
- Compares VADER compound scores vs TextBlob polarity
- Analyzes correlation between methods

### Test 2: Feature Engineering
- Creates additional features:
  - Word count, average word length
  - Exclamation/question mark counts
  - Capital letter ratio
  - Day of week encoding

### Test 3: Multiple ML Models Comparison
- Tests 4 different algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- Compares ROC-AUC, accuracy, and training time

### Test 4: Cross-Validation Testing
- 5-fold stratified cross-validation
- Robust model evaluation

### Test 5: Feature Importance Analysis
- Identifies most important features using Random Forest

### Test 6: Statistical Tests
- ANOVA test for sentiment across ratings
- Correlation tests
- T-tests for time-of-day differences

### Test 7: Performance Visualizations
- ROC curves for all models
- Model comparison charts

### Test 8: Confusion Matrix Analysis
- Detailed prediction analysis
- Precision, recall, specificity, F1-score

### Test 9: Feature Set Comparison
- Tests different feature combinations
- Identifies optimal feature set

### Test 10: Comprehensive Summary Report
- Complete overview of all test results
- Recommendations and findings

## 📈 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "Week 2/VSCodeExperiment.ipynb"
```

2. Run cells sequentially:
   - Cells 0-11: Data loading and preprocessing
   - Cell 12: Testing framework setup
   - Cells 13-21: All test cells
   - Cell 22: Summary report

## 📝 Key Findings

(Add your key findings here after running the analysis)

## 🛠️ Technologies Used

- **Python**: Data processing and analysis
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models
- **VADER Sentiment**: Sentiment analysis
- **TextBlob**: Alternative sentiment analysis
- **Matplotlib/Seaborn**: Visualizations
- **SciPy**: Statistical tests

## 📊 Results

(Add your results summary here)

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

This project is for academic purposes.

## 👤 Author

Abdullah Ahmad Adel Al-Taher
- Course: GRAD699
- Institution: Harrisburg University of Science and Technology

## 🙏 Acknowledgments

- Amazon for providing review data
- Open-source community for excellent libraries

