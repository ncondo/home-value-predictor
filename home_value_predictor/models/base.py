"""Model class to be extended by model-specific classes."""
from pathlib import Path


class Model:
    """Simple abstract class for models."""

    @classmethod
    def model_dirname(cls):
        return Path(__file__).resolve().parents[0] / "saved_models"


    def load_model(self):
        pass


    def save_model(self):
        pass


    def train_model(self):
        pass


    def predict(self):
        pass