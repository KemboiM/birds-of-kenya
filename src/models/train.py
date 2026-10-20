import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

MODEL_PATH = 'models/best_model.keras'


def build_model(num_classes, input_shape=(224, 224, 3), weights='imagenet', dropout=0.3):
    """MobileNetV2 transfer-learning classifier with a softmax head.

    Inputs are raw 0-255 RGB images; augmentation and MobileNetV2 scaling
    ([-1, 1]) happen inside the model, so the saved model accepts plain images.
    """
    augment = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name='augmentation')

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name='mobilenet_scaling')(x)
    x = base_model(x, training=False)   # keep BatchNorm in inference mode, also when fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def _compile(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')],
    )


def _callbacks(patience):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                          patience=2, verbose=1),
    ]


def train_model(train_ds, val_ds, num_classes, epochs=10, fine_tune_epochs=10,
                fine_tune_layers=40, weights='imagenet', patience=4, img_size=224):
    """Two-phase training.

    1. Train only the new head with the MobileNetV2 base frozen.
    2. Unfreeze the top `fine_tune_layers` of the base and continue at a low
       learning rate. Set fine_tune_epochs=0 to skip.
    """
    model, base_model = build_model(num_classes, input_shape=(img_size, img_size, 3),
                                    weights=weights)

    print("\n=== Phase 1: training classifier head ===")
    _compile(model, 1e-3)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=_callbacks(patience), verbose=1)

    if fine_tune_epochs > 0:
        print(f"\n=== Phase 2: fine-tuning top {fine_tune_layers} MobileNetV2 layers ===")
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        _compile(model, 1e-5)
        start = len(history.history['loss'])
        fine = model.fit(train_ds, validation_data=val_ds,
                         epochs=start + fine_tune_epochs, initial_epoch=start,
                         callbacks=_callbacks(patience), verbose=1)
        for key, values in fine.history.items():
            history.history.setdefault(key, []).extend(values)

    return model, history


def evaluate_model(model, test_ds, class_names):
    """Accuracy, top-5 accuracy, macro precision/recall/F1 and a per-species report."""
    y_true, probs = [], []
    for images, labels in test_ds:
        probs.append(model.predict(images, verbose=0))
        y_true.append(labels.numpy())
    probs = np.concatenate(probs)
    y_true = np.concatenate(y_true)
    y_pred = probs.argmax(axis=1)

    k = min(5, len(class_names))
    top_k = np.argsort(probs, axis=1)[:, -k:]
    top5 = float(np.mean([t in row for t, row in zip(y_true, top_k)]))

    labels = list(range(len(class_names)))
    metrics = {
        'num_classes': len(class_names),
        'test_images': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        f'top{k}_accuracy': top5,
        'precision_macro': float(precision_score(y_true, y_pred, labels=labels,
                                                 average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, labels=labels,
                                           average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, labels=labels,
                                   average='macro', zero_division=0)),
    }
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    print("\n=== Test Results ===")
    for key, value in metrics.items():
        print(f"{key:<16} {value:.4f}" if isinstance(value, float) else f"{key:<16} {value}")

    worst = sorted(class_names, key=lambda c: report[c]['f1-score'])[:5]
    print("\nHardest species (lowest F1):")
    for name in worst:
        print(f"  {name:<35} F1 {report[name]['f1-score']:.2f}  (n={int(report[name]['support'])})")

    return metrics, report
