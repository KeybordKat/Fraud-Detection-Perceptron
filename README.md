# Fraud Detection Perceptron Model

A machine learning project that transforms a custom perceptron algorithm into a practical fraud detection system for credit card transactions.

## Overview

This project evolved from a basic perceptron implementation with randomized mock data into a comprehensive fraud detection tool using real-world credit card transaction data. The system addresses the challenging problem of class imbalance in fraud detection while optimizing for business-relevant metrics.

## Dataset

The project uses credit card transaction data from Kaggle: [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/kelvinobiri/credit-card-transactions)

**Key Challenge:** Fraudulent transactions represent less than 0.2% of the dataset, creating a severe class imbalance problem.

## Development Progress

### Initial Implementation
Starting with the original perceptron algorithm, I replaced mock data with real transaction data from Kaggle.

**Initial Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Legit (-1) | 1.00 | 1.00 | 1.00 | 59,915 |
| Fraud (+1) | 0.46 | 0.60 | 0.52 | 85 |
| **Accuracy** | | | **1.00** | **60,000** |

*Result: Good performance on legitimate transactions, but poor fraud detection due to class imbalance.*

### SMOTE Implementation
Applied Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance, as recommended in the dataset documentation.

**Performance After SMOTE:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Legit (-1) | 1.00 | 0.76 | 0.86 | 59,915 |
| Fraud (+1) | 0.01 | 1.00 | 0.01 | 85 |
| **Accuracy** | | | **0.76** | **60,000** |

*Result: Perfect fraud recall achieved, but accuracy dropped due to perceptron limitations with noisy boundaries.*

### Feature Engineering
Created custom features from raw transaction data to improve model performance:

- **Account balance changes** for senders and recipients
- **Transaction amount** compared to before/after balance ratios
- **Pattern indicators** for suspicious transaction behavior

**Performance After Feature Engineering:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Legit (-1) | 1.00 | 0.92 | 0.96 | 59,915 |
| Fraud (+1) | 0.02 | 1.00 | 0.04 | 85 |
| **Accuracy** | | | **0.76** | **60,000** |

*Result: Improved legitimate transaction recall while maintaining perfect fraud detection.*

### Threshold Optimization
Implemented dynamic decision threshold using precision-recall curve analysis:

- **Goal:** Maintain 90% fraud recall while maximizing precision
- **Method:** Cost-sensitive learning to reduce false alarms
- **Result:** Stable performance between epochs 30-40

### Technology Stack Upgrade
Migrated from basic Python scripts to Jupyter Notebook for:
- Better documentation and visualization
- Enhanced development workflow
- Improved result presentation

## Key Features

- **Custom Perceptron Implementation**: Built from scratch algorithm
- **Class Imbalance Handling**: SMOTE oversampling technique
- **Feature Engineering**: Domain-specific feature creation
- **Dynamic Thresholding**: Precision-recall curve optimization
- **Cost-Sensitive Learning**: Business-oriented metric optimization

## Business Impact

The fraud detection system prioritizes:
- **High Recall**: Catching fraudulent transactions (100% fraud recall achieved)
- **Optimized Precision**: Minimizing false alarms to reduce customer inconvenience
- **Cost Awareness**: Understanding that missing fraud is more expensive than false positives

## Technical Implementation

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install imbalanced-learn  # for SMOTE
```

### Project Structure
```
fraud-detection-perceptron/
├── data/
│   └── credit_card_transactions.csv
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── src/
│   └── perceptron.py
├── images/
│   ├── graph_1.png
│   ├── graph_2.png
│   └── graph_3.png
└── README.md
```

## Results Visualization

The project includes comprehensive visualizations showing:
- **Before/After Feature Engineering**: Model separation improvement
- **Precision-Recall Curves**: Threshold optimization analysis
- **Performance Metrics**: Training stability across epochs

## Lessons Learned

1. **Class Imbalance**: Critical challenge in fraud detection requiring specialized techniques
2. **Feature Engineering**: Domain knowledge significantly improves model performance
3. **Business Metrics**: Recall prioritization over precision in fraud detection scenarios
4. **Threshold Tuning**: Dynamic decision boundaries improve practical performance

## Future Improvements

- Ensemble methods for better accuracy
- Real-time prediction pipeline
- Advanced feature engineering techniques
- Integration with production systems

## License

This project is available under the MIT License.

## Contact

Feel free to reach out for questions or collaboration opportunities!

---

*This project demonstrates practical machine learning problem-solving, from handling real-world data challenges to optimizing for business-relevant metrics in fraud detection.*
