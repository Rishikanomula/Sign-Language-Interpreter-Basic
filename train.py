# models/train.py
import torch, numpy as np, argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from lstm_model import LSTMClassifier

def train_loop(X, y, input_size, n_classes, epochs=20, batch=16, lr=1e-3):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long()), batch_size=batch)
    model = LSTMClassifier(input_size, hidden_size=256, num_layers=2, num_classes=n_classes, bidirectional=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for ep in range(epochs):
        model.train()
        tot, acc = 0.0, 0
        for xb, yb in tr_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        # validation
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds==yb).sum().item()
                total += yb.size(0)
        print(f"Epoch {ep+1}/{epochs} loss={tot/len(tr_loader):.4f} val_acc={correct/total:.3f}")
    torch.save(model.state_dict(), "lstm_sign.pth")
    print("Saved model lstm_sign.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    X = np.load(f"{args.data_prefix}_X.npy")  # (N, seq, feat)
    y = np.load(f"{args.data_prefix}_y.npy")
    input_size = X.shape[2]
    n_classes = len(set(y))
    train_loop(X, y, input_size, n_classes, epochs=args.epochs)
