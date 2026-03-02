from tensorflow.keras.models import load_model

def load_trained_model(path="models/lstm_autoencoder.h5"):
    return load_model(path)

def compute_reconstruction_error(model, X):
    X_recon = model.predict(X)
    return ((X - X_recon) ** 2).mean(axis=(1,2))
