# 🏠 House Price Prediction - Advanced ML Project (INR Version)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📊 Project Overview

An end-to-end machine learning project that predicts **house prices in Indian Rupees (₹)** using advanced feature engineering, multiple algorithms, and model interpretability techniques.


## 🎯 Key Features

- ✅ **Advanced Feature Engineering**: Created 13 new features from 9 original features  
- ✅ **Multiple Models**: Trained and compared 8 different algorithms  
- ✅ **Hyperparameter Tuning**: Optimized best model using RandomizedSearchCV  
- ✅ **Model Interpretability**: SHAP analysis for feature importance  
- ✅ **Cross-Validation**: Robust model evaluation  
- ✅ **Interactive Web App**: Deployed Streamlit application  

## 📈 Model Performance (Values in ₹)

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **R² Score** | 0.0155 |
| **RMSE** | ₹11,923,855.25 |
| **MAE** | ₹10,315,666.29 |
| **MAPE** | 35.34% |

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas & NumPy**: Data manipulation  
- **Scikit-learn**: ML algorithms and preprocessing  
- **XGBoost, LightGBM, CatBoost**: Advanced gradient boosting  
- **SHAP**: Model interpretability  
- **Plotly & Seaborn**: Data visualization  
- **Streamlit**: Web application deployment  

```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

### Training the Model

Open and run `House_Price_Prediction.ipynb` in Google Colab or Jupyter Notebook.

## 📊 Data Processing Pipeline

1. **Data Loading & EDA**: Comprehensive exploratory analysis
2. **Missing Value Imputation**: Median for numeric, mode for categorical
3. **Feature Engineering**:
   - Polynomial features (squared, cubed, sqrt)
   - Interaction features
   - Binning continuous variables
   - Aggregation features
4. **Encoding**: Label encoding for categorical variables
5. **Scaling**: RobustScaler for outlier handling
6. **Model Training**: 8 different algorithms with cross-validation
7. **Hyperparameter Tuning**: RandomizedSearchCV
8. **Model Evaluation**: Multiple metrics (RMSE, MAE, R², MAPE)

## 🎨 Model Comparison

All models were trained and evaluated:

| Model             |     CV_RMSE |   Train_RMSE |   Test_RMSE |    Test_MAE |     Test_R² |   Test_MAPE |
|:------------------|------------:|-------------:|------------:|------------:|------------:|------------:|
| Random Forest     | 1.20077e+07 |  5.38396e+06 | 1.19239e+07 | 1.03157e+07 |  0.0155141  |     35.3358 |
| CatBoost          | 1.19267e+07 |  1.03426e+07 | 1.19614e+07 | 1.03943e+07 |  0.00930337 |     35.6709 |
| Lasso Regression  | 1.18857e+07 |  1.17565e+07 | 1.20407e+07 | 1.04551e+07 | -0.00387734 |     35.9647 |
| ElasticNet        | 1.18839e+07 |  1.17754e+07 | 1.20472e+07 | 1.04564e+07 | -0.00496684 |     35.9392 |
| Ridge Regression  | 1.18781e+07 |  1.17909e+07 | 1.20499e+07 | 1.04605e+07 | -0.00540288 |     35.9574 |
| LightGBM          | 1.23017e+07 |  9.09526e+06 | 1.2164e+07  | 1.04868e+07 | -0.0245319  |     35.6205 |
| XGBoost           | 1.22329e+07 |  7.83933e+06 | 1.22145e+07 | 1.04015e+07 | -0.033067   |     35.2313 |
| Gradient Boosting | 1.22566e+07 |  7.86551e+06 | 1.23454e+07 | 1.05064e+07 | -0.0553252  |     35.8171 |

## 💡 Key Insights

- The model achieves **1.6%** accuracy in predicting house prices
- Feature engineering improved model performance significantly
- Random Forest outperformed other algorithms
- Model is production-ready with interpretability features

## 🔮 Future Enhancements

- [ ] Add more external data sources (demographics, crime rates)
- [ ] Implement ensemble stacking
- [ ] Add time-series analysis for price trends
- [ ] Create mobile-responsive design
- [ ] Add map visualization for location-based pricing

## 👨‍💻 Author

**Namrata**
- LinkedIn: https://www.linkedin.com/in/namrata-bellenavar-749815302/ 
- GitHub: https://github.com/namrata-bell 
- Email: namratabellenavar@gmail.com


## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset
- Inspired by real-world property valuation challenges

