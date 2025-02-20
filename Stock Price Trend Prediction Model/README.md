# Stock Price Prediction using LSTM

## Introduction
This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The dataset consists of historical stock prices, including Open, High, Low, Close, and Volume data, specifically for the **S&P 500 index on a minute-by-minute basis**. The goal is to train an LSTM model that can accurately forecast future stock prices based on past trends.

## Dataset
- The dataset contains minute-level S&P 500 stock price data with timestamps.
- It includes various technical indicators such as Moving Averages (MA7, MA21), Exponential Moving Average (EMA), Relative Strength Index (RSI), and Volatility.
- Missing values were checked and handled appropriately.
- The dataset is large and requires careful memory management for processing.

## Methodology
### Data Preprocessing
- The dataset was cleaned, and minute-level technical indicators were calculated to enhance feature engineering.
- Data was split into training and test sets.
- Normalization was applied to scale the data before feeding it into the LSTM model.
- Sequence lengths were carefully chosen to capture minute-by-minute trends effectively.

### Model Architecture
- The LSTM model consists of:
  - Multiple stacked LSTM layers to capture sequential dependencies.
  - Fully connected (Dense) layers for final predictions.
  - Dropout layers to prevent overfitting.
  - Batch normalization to stabilize training.
- The model was trained using the Adam optimizer with Mean Squared Error (MSE) as the loss function.
- Early stopping was implemented to prevent overfitting during training.

### Evaluation & Results
- The model was evaluated using test data, achieving an accuracy of **60%**.
- The ARIMA model was also used for comparison, but it struggled to capture long-term stock trends at the minute level.
- The plot (image.png) illustrates that the LSTM model captures trends but has limitations in precise forecasting.
- The evaluation includes Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) to assess performance.

## Observations
- The model successfully learns short-term trends but may struggle with long-term predictions due to minute-level volatility.
- The high-frequency nature of the data introduces noise, which impacts model stability.
- Further improvements could include tuning hyperparameters, incorporating external financial indicators, or trying different sequence lengths.
- Alternative deep learning models, such as Transformers or hybrid LSTM-CNN models, could be explored for better performance.

## Usage
1. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn
   ```
2. Run the Jupyter Notebook (`Lstm.ipynb`) to preprocess the data, train the model, and evaluate results.
3. Adjust hyperparameters and model architecture for further improvements.
4. Use the trained model to make real-time minute-level stock predictions.

## Future Improvements
- Experimenting with different LSTM architectures and hyperparameters.
- Using additional financial indicators or sentiment analysis.
- Comparing with other deep learning models like Transformers.
- Handling market anomalies and sudden fluctuations in high-frequency data.
- Implementing reinforcement learning for improved trading strategies.

This project provides a foundation for stock price forecasting using deep learning. Improvements in feature engineering, model tuning, and external data sources could enhance predictive performance.

