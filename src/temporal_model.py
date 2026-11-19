import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

INPUT_FEATURES = 12
SEQUENCE_LENGTH = 16
GRU_HIDDEN_SIZE = 64
GRU_DROPOUT = 0.3
FC_HIDDEN_SIZE = 32
OUTPUT_SIZE = 1

RISK_THRESHOLDS = {
    "normal": 0.33,
    "elevated": 0.66,
    "high": 0.85,
}


class GRUModel(nn.Module):
    def __init__(self, input_features=INPUT_FEATURES, hidden_size=GRU_HIDDEN_SIZE, dropout=GRU_DROPOUT, fc_hidden_size=FC_HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_hidden_size = fc_hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True, dropout=dropout if True else 0)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)
        last_out = gru_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc2(out)
        prob = self.sigmoid(out)
        return prob, hidden


class LSTMModel(nn.Module):
    def __init__(self, input_features=INPUT_FEATURES, hidden_size=GRU_HIDDEN_SIZE, dropout=GRU_DROPOUT, fc_hidden_size=FC_HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True, dropout=dropout if True else 0)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc2(out)
        prob = self.sigmoid(out)
        return prob, hidden


def initialize_model(model_type="gru", device="cpu"):
    if model_type.lower() == "gru":
        model = GRUModel()
    elif model_type.lower() == "lstm":
        model = LSTMModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


def compute_loss(output, target):
    bce = nn.BCEWithLogitsLoss()
    loss = bce(output.squeeze(), target.float())
    return loss


def predict_risk(probability):
    if probability >= RISK_THRESHOLDS["high"]:
        return "high"
    elif probability >= RISK_THRESHOLDS["elevated"]:
        return "elevated"
    else:
        return "normal"


def get_risk_explanation(features, probability):
    explanations = []
    if probability > 0.5:
        explanations.append("Elevated model risk score")
    text = "Risk factors: " + "; ".join(explanations) if explanations else "No specific risk factors detected"
    text += f" | Model probability: {probability:.2f}"
    return text


def test_model():
    model = GRUModel()
    model.eval()
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_FEATURES)
    with torch.no_grad():
        output, hidden = model(dummy_input)
    print(f"GRU model test: output shape {output.shape}, hidden shape {hidden.shape}")
    print(f"Output probability: {output.item():.4f}")
    lstm_model = LSTMModel()
    lstm_model.eval()
    with torch.no_grad():
        lstm_output, lstm_hidden = lstm_model(dummy_input)
    print(f"LSTM model test: output shape {lstm_output.shape}")
    risk_level = predict_risk(0.9)
    print(f"Risk level for 0.9: {risk_level}")
    risk_level2 = predict_risk(0.4)
    print(f"Risk level for 0.4: {risk_level2}")
    features = np.array([1.0, 1.0, 0.0, 0.8, 0.01, 500.0, 200.0, 10.0, 15.0, 0.1, 0.5, 2.0])
    explanation = get_risk_explanation(features, 0.9)
    print(f"Risk explanation: {explanation}")
    return True


if __name__ == "__main__":
    test_model()