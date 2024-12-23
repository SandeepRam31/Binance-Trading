# Real-Time Cryptocurrency Prediction with Continuous Learning

## Overview
This project demonstrates a deep learning-based system for predicting cryptocurrency price movements in real-time using Binance WebSocket data. The system uses historical data for initial training and continuously learns from live data to adapt to market trends.

## Features
- **Deep Learning Model**: A neural network predicts whether the price will move up or down in the next 30 seconds.
- **Historical Data Training**: The model is pre-trained using one month of historical data loaded from a CSV file.
- **Continuous Learning**: The model updates its weights every 30 seconds based on live price data and prediction accuracy.
- **Real-Time Feedback**: Displays prediction outcomes in green (correct) or red (incorrect) alongside live Binance kline data.
- **KL Divergence Regularization**: Ensures gradual weight updates to prevent overfitting to recent data.

## Requirements
- Python 3.7+
- Binance API key and secret
- Libraries: `python-binance`, `pandas`, `torch`, `termcolor`

Install dependencies:
```bash
pip install python-binance pandas torch termcolor
```

## How It Works
1. **Historical Data Loading**:
   - Historical OHLCV data is loaded from a CSV file (`historical_data.csv`) with columns: `timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`.
   - The `Close` column is used to train the initial model.

2. **Real-Time Prediction**:
   - The system connects to Binance WebSocket to fetch real-time kline data (1-second interval).
   - The model predicts price movement every 30 seconds based on the last 60 seconds of data.

3. **Feedback and Learning**:
   - The model compares its prediction with actual price changes after 30 seconds.
   - Correct predictions are displayed in green; incorrect predictions in red.
   - Loss is calculated using cross-entropy and KL divergence to update the model.

## Code Structure
- **Neural Network**: A feedforward neural network (`PricePredictor`) with a ReLU activation and softmax output.
- **Historical Data Training**: `train_on_historical_data` trains the model on preloaded CSV data.
- **Real-Time Processing**: WebSocket callback (`handle_socket_message`) handles live data, generates predictions, and updates the model.

## Usage
1. Prepare a CSV file (`historical_data.csv`) with one month of cryptocurrency data in the required format.
2. Set your Binance API key and secret in the code:
   ```python
   api_key = '<api_key>'
   api_secret = '<api_secret>'
   ```
3. Run the script:
   ```bash
   python script_name.py
   ```

4. Observe predictions and feedback in the console:
   - **Green**: Correct predictions
   - **Red**: Incorrect predictions

## Example Console Output
```
Gathering data... (59 / 60)
Predicted Signal: UP
Kline Message: {<kline data>} [displayed in green/red based on correctness]
```

## Future Enhancements
- Incorporate more advanced features such as technical indicators or additional data sources.
- Improve model architecture using recurrent networks (e.g., LSTM or GRU) for better sequential data handling.
- Deploy the system for automated trading.

## Disclaimer
This project is for educational purposes only and should not be used for live trading without proper testing and risk management.

