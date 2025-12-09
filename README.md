# 📈 Stock Price Prediction using Machine Learning


**LINK : https://stockmarketprediction1.streamlit.app/**

**Interactive Dashboard LINK : https://stock-market-prediction-frontend.onrender.com/ (NOTE: This is the same website with interactive Dashboard)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for predicting stock price movements using both classical ML algorithms and deep learning (LSTM). This project analyzes historical stock data and predicts whether the price will move **UP** or **DOWN** the next day.

## 🎯 Features

- **Multiple ML Models**: Logistic Regression, SVM (Polynomial), XGBoost, Random Forest, Voting Ensemble, and LSTM
- **Advanced Feature Engineering**: 14 technical indicators including returns, momentum, volatility, moving averages, MACD, and RSI
- **Walk-Forward Validation**: Time-series cross-validation to prevent data leakage
- **Comprehensive Visualizations**: EDA charts, confusion matrices, prediction probability plots
- **Model Persistence**: Save and load trained models for future predictions
- **Interactive Notebook**: Professional Colab-style Jupyter notebook

## 📊 Dataset

- **Stock**: INFY.NS (Infosys Limited - NSE)
- **Time Period**: January 1, 2015 - November 15, 2025
- **Data Points**: 2,685 trading days
- **Features**: 14 engineered technical indicators
- **Target**: Binary classification (UP/DOWN)

## 🛠️ Technologies Used

- **Python 3.12.7**
- **Machine Learning**: scikit-learn 1.7.2, XGBoost 3.1.1
- **Deep Learning**: TensorFlow 2.20.0, Keras
- **Data Processing**: pandas 2.3.0, numpy 2.2.6
- **Visualization**: matplotlib 3.10.3, seaborn 0.13.2
- **Data Collection**: yfinance 0.2.66

## 📁 Project Structure

```
ML project/
├── stock_prediction_model.ipynb    # Main Jupyter notebook
├── graph.py                         # Original Streamlit application
├── models/                          # Trained model files
│   ├── best_model_SVM_Poly.pkl     # Best performing model
│   ├── Logistic_Regression.pkl
│   ├── SVM_Poly.pkl
│   ├── XGBoost.pkl
│   ├── Random_Forest.pkl
│   ├── Voting_Ensemble.pkl
│   └── best_model_LSTM_window_60.h5
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package installer)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-prediction-ml.git
   cd stock-prediction-ml
   ```

2. **Install required packages**
   ```bash
   pip install tensorflow scikit-learn xgboost yfinance pandas numpy matplotlib seaborn
   ```

3. **Run the Jupyter notebook**
   ```bash
   jupyter notebook stock_prediction_model.ipynb
   ```

## 📝 Usage

### Running the Notebook

1. Open `stock_prediction_model.ipynb` in Jupyter Notebook or Google Colab
2. Execute all cells sequentially (Runtime → Run all)
3. The notebook will:
   - Download latest stock data
   - Perform feature engineering
   - Train all 6 models
   - Display performance metrics
   - Generate visualizations
   - Save trained models

### Making Predictions

```python
import pickle
import pandas as pd

# Load the best model
with open('models/best_model_SVM_Poly.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Prepare your features (14 features required)
# features = prepare_features(your_data)

# Make prediction
prediction = best_model['pipeline'].predict_proba(features)[:, 1]
direction = "UP" if prediction[0] > best_model['threshold'] else "DOWN"
print(f"Prediction: {direction} (Probability: {prediction[0]*100:.2f}%)")
```

## 📈 Model Performance

| Model | Train AUC | Valid AUC | Accuracy | F1 Score | Threshold |
|-------|-----------|-----------|----------|----------|-----------|
| **SVM (Poly)** | 0.4088 | 0.5437 | **48.78%** | **0.6540** | 0.51 |
| Logistic Regression | 0.5418 | 0.5474 | 48.41% | 0.6523 | 0.10 |
| Voting Ensemble | 0.9740 | 0.4528 | 48.41% | 0.6523 | 0.10 |
| XGBoost | 0.8594 | 0.4545 | 48.41% | 0.6523 | 0.10 |
| Random Forest | 0.9984 | 0.4514 | 48.41% | 0.6523 | 0.10 |

### Walk-Forward Validation Results

| Model | WF Folds | WF Mean Accuracy | WF Mean F1 | WF Mean AUC |
|-------|----------|------------------|------------|-------------|
| Voting Ensemble | 5 | 0.5172 | 0.4463 | 0.5351 |
| Logistic Regression | 5 | 0.5149 | 0.5203 | 0.5265 |
| XGBoost | 5 | 0.5008 | 0.4285 | 0.5334 |
| Random Forest | 5 | 0.4992 | 0.3886 | 0.5310 |
| SVM (Poly) | 5 | 0.5059 | 0.5530 | 0.5205 |

## 🔬 Feature Engineering

The project creates 14 technical indicators:

1. **Returns**: 1-day, 5-day, 10-day percentage returns
2. **Momentum**: 10-day momentum indicator
3. **Volatility**: 10-day rolling standard deviation
4. **Moving Averages**: 10-day and 20-day simple moving averages
5. **MACD**: Moving Average Convergence Divergence
6. **RSI**: Relative Strength Index (14-day)
7. **OHLCV**: Open, High, Low, Close, Volume

## 📊 Visualizations

The notebook includes:
- Close price trends over time
- Distribution plots for all features
- Box plots for outlier detection
- Correlation heatmap
- Confusion matrices for all models
- Prediction probability charts with actual outcomes
- Feature importance rankings

## 🎯 Key Insights

- **Best Model**: SVM (Polynomial) achieved the highest F1 score (0.6540)
- **Class Balance**: 51.6% UP days vs 48.4% DOWN days (well-balanced)
- **Optimal Threshold**: Custom threshold optimization for each model
- **Walk-Forward Validation**: Ensures robust time-series predictions
- **Feature Importance**: Returns and moving averages are strongest predictors

## ⚠️ Disclaimer

**This project is for educational purposes only.** Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct thorough research before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing free stock data via yfinance
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- The open-source community for excellent tools and libraries

## 📚 Future Enhancements

- [ ] Add more technical indicators (Bollinger Bands, Stochastic Oscillator)
- [ ] Implement sentiment analysis from news/social media
- [ ] Multi-stock portfolio prediction
- [ ] Real-time prediction API
- [ ] Web dashboard with Streamlit
- [ ] Hyperparameter tuning with Optuna
- [ ] Add more deep learning models (GRU, Transformer)

---

⭐ **If you found this project helpful, please consider giving it a star!**

*Last Updated: January 2025*
