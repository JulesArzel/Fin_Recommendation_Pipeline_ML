# Stock Recommendation System

This project implements a pipeline for generating BUY, SELL, or HOLD recommendations on selected companies by combining multiple analytical approaches.  

The system integrates traditional financial metrics, machine learning models, sentiment analysis, and deep learning forecasts to provide aggregated investment signals.

## Features

- **Data Collection**  
  Fetches company ratios and financial data from various sources.

- **Clustering**  
  Groups companies based on financial performance and ratios.

- **Classification**  
  Predicts investment signals (BUY, HOLD, SELL) using supervised models.

- **Regression Models**  
  Applies models such as Random Forest and XGBoost for performance forecasting.

- **Sentiment Analysis**  
  Analyzes market sentiment from textual data.

- **Deep Learning Forecasting**  
  Uses an LSTM model to forecast company stock prices and evaluate predictive accuracy.

- **Signal Aggregation**  
  Final recommendation is derived from a combination of:
  - Classifier prediction  
  - Sentiment polarity  
  - Comparison of LSTM vs XGBoost forecasting errors  

  Strategy:  
  - **BUY** if classifier and sentiment are positive, and LSTM outperforms XGBoost  
  - **SELL** if classifier and sentiment are negative, and LSTM outperforms XGBoost  
  - **HOLD** otherwise  

## Technologies Used

- Python 3  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras (for LSTM models)  
- Pandas, NumPy  
- Matplotlib  

## Project Structure

