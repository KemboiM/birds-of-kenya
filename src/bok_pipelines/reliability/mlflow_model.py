"""MLflow model for a saved Keras image classifier, so it can be registered, given an alias such
as "production", and loaded back with mlflow.pyfunc.load_model on any computer.

MLflow stores this file with the model ("models from code"), so loading it does not need the
rest of this repository. Input: a DataFrame with a column ``path`` (image files).
Output: one row for each image with label, confidence and the top 3 labels.
"""

import mlflow
import mlflow.pyfunc
import pandas as pd


class KerasImageClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import keras
        self.model = keras.models.load_model(context.artifacts["keras_model"])
        with open(context.artifacts["labels"]) as f:
            self.labels = [line.strip() for line in f if line.strip()]
        self.size = self.model.input_shape[1]

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        import numpy as np
        from PIL import Image, ImageOps

        paths = list(model_input["path"])
        batch = np.stack([
            np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((self.size, self.size)),
                       dtype="float32")
            for p in paths])                                   # raw 0-255; the model does its own scaling
        probs = self.model.predict(batch, verbose=0)
        top = np.argsort(probs, axis=1)[:, ::-1][:, :3]
        return pd.DataFrame({
            "path": paths,
            "label": [self.labels[t[0]] for t in top],
            "confidence": [float(probs[i, t[0]]) for i, t in enumerate(top)],
            "top3": ["; ".join(f"{self.labels[j]} {probs[i, j]:.2f}" for j in t) for i, t in enumerate(top)],
        })


mlflow.models.set_model(KerasImageClassifier())
