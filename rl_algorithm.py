import time
from collections import deque
from binance import ThreadedWebsocketManager
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import colored

# Neural Network for Price Prediction
class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Outputs: [Price Down, Price Up]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

api_key = '<api_key>'
api_secret = '<api_secret>'

# Hyperparameters
input_size = 60  # Last 60 seconds of price data
hidden_size = 128
learning_rate = 0.001
kl_lambda = 0.1  # Weight for KL divergence loss
batch_size = 1

# Initialize Model, Optimizer, and Loss Functions
model = PricePredictor(input_size=input_size, hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Store historical data for initial training
historical_prices = deque(maxlen=60 * 60 * 24 * 30)

# Live price queue for continuous prediction
live_prices = deque(maxlen=input_size)

# Load historical data from CSV
def load_historical_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['Close'].values

# Function to train model on historical data
def train_on_historical_data(historical_data):
    data = torch.tensor(historical_data, dtype=torch.float32)
    for i in range(len(data) - input_size - 30):
        x = data[i:i + input_size].unsqueeze(0)
        y = torch.tensor([1 if data[i + input_size + 30] > data[i + input_size] else 0]).unsqueeze(0)

        # Forward pass
        outputs = model(x)
        loss = loss_function(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    symbol = 'BTCUSDT'
    previous_signal_correct = None

    # Load historical data and train the model
    historical_data = load_historical_data('historical_data.csv')
    train_on_historical_data(historical_data)

    def handle_socket_message(msg):
        nonlocal previous_signal_correct

        if msg['e'] == 'kline':
            kline = msg['k']
            close_price = float(kline['c'])
            live_prices.append(close_price)

            if len(live_prices) < input_size:
                print(f"Gathering data... ({len(live_prices)} / {input_size})")
                return

            # Predict every 30 seconds
            if len(live_prices) % 30 == 0:
                x = torch.tensor(list(live_prices), dtype=torch.float32).unsqueeze(0)
                predictions = model(x)
                signal = "UP" if torch.argmax(predictions) == 1 else "DOWN"

                # Display prediction
                print(f"Predicted Signal: {signal}")

                # Wait 30 seconds to collect actual data for loss computation
                time.sleep(30)

                actual_price = live_prices[-1]
                label = torch.tensor([1 if actual_price > live_prices[-31] else 0]).unsqueeze(0)

                # Check prediction correctness
                is_correct = (torch.argmax(predictions) == label.item())
                previous_signal_correct = is_correct

                # Display colored result
                result_color = "green" if is_correct else "red"
                print(colored(f"Kline Message: {msg}", result_color))

                # Compute KL divergence loss
                prev_predictions = predictions.detach()
                current_predictions = model(x)
                kl_loss = kl_lambda * torch.sum(prev_predictions * torch.log(prev_predictions / current_predictions))

                # Total loss
                classification_loss = loss_function(current_predictions, label)
                total_loss = classification_loss + kl_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    # Initialize WebSocket
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()
    twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval='1s')

    try:
        twm.join()
    except KeyboardInterrupt:
        print("Stopping...")
        twm.stop()

if __name__ == "__main__":
    main()
