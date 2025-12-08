# inference/infer.py
import numpy as np, torch, joblib
from models.lstm_model import LSTMClassifier

class Predictor:
    def __init__(self, model_path, labelenc_path, seq_len=20):
        self.le = joblib.load(labelenc_path)
        data = np.load("data_X.npy")  # load to get dims or pass dims
        input_size = data.shape[2]
        n_classes = len(self.le.classes_)
        self.model = LSTMClassifier(input_size, hidden_size=256, num_layers=2, num_classes=n_classes, bidirectional=True)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.seq_len = seq_len
        self.buffer = []

    def push_keypoints(self, kp_vector):
        if kp_vector is None:
            kp_vector = np.zeros((self.model.lstm.input_size,), dtype=np.float32)
        self.buffer.append(kp_vector)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def predict(self):
        if len(self.buffer) < self.seq_len:
            return None
        X = np.stack(self.buffer)[None,:,:]  # (1,seq,feat)
        with torch.no_grad():
            out = self.model(torch.tensor(X).float())
            p = torch.softmax(out, dim=1).numpy()[0]
            idx = p.argmax()
            return self.le.inverse_transform([idx])[0], p.max()
