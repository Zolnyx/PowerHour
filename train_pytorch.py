import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ======================
# 1. Data Loading & Preparation
# ======================
def load_data():
    # Load input vectors (skip first 2 columns - video ID and frame number)
    df_inputs = pd.read_csv("data/input_vectors.csv", header=None)
    X = df_inputs.iloc[:, 2:7].values  # Columns 2-6 contain the 5 features
    
    # Load labels from output_vectors.csv (generated from labels.csv)
    df_outputs = pd.read_csv("data/output_vectors.csv", header=None)
    y = df_outputs.iloc[:, 2:8].values  # Columns 2-7 contain [c, k, h, r, x, i]
    
    # Convert to numeric, handling any non-numeric values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
    y = pd.DataFrame(y).apply(pd.to_numeric, errors='coerce').values
    
    # Remove rows with NaN values
    valid_rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 2. Model Architecture
# ======================
class SquatPostureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 64),  # 5 input features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6)   # 6 output classes [c, k, h, r, x, i]
        )
        
    def forward(self, x):
        return self.net(x)

# ======================
# 3. Training Execution
# ======================
def train():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Create DataLoader
    train_data = TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SquatPostureModel()
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(200):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        if epoch % 20 == 0:
            with torch.no_grad():
                test_preds = model(torch.from_numpy(X_test))
                test_loss = criterion(test_preds, torch.from_numpy(y_test))
                print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), "squat_model.pth")
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    train()