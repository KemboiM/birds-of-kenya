import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


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


def _callbacks(patience, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                          patience=2, verbose=1),
    ]


def train_model(train_ds, val_ds, num_classes, epochs=10, fine_tune_epochs=10,
                fine_tune_layers=40, weights='imagenet', patience=4, img_size=224,
                model_path='model/model_assets/trained_checkpoints/birds_classify.keras'):
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
                        callbacks=_callbacks(patience, model_path), verbose=1)

    if fine_tune_epochs > 0:
        print(f"\n=== Phase 2: fine-tuning top {fine_tune_layers} MobileNetV2 layers ===")
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        _compile(model, 1e-5)
        start = len(history.history['loss'])
        fine = model.fit(train_ds, validation_data=val_ds,
                         epochs=start + fine_tune_epochs, initial_epoch=start,
                         callbacks=_callbacks(patience, model_path), verbose=1)
        for key, values in fine.history.items():
            history.history.setdefault(key, []).extend(values)

    return model, history


def _feature_extractor(img_size, weights):
    """Raw 0-255 image -> 1280 MobileNetV2 features (same path as build_model, minus the head)."""
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base(x, training=False)
    return keras.Model(inputs, layers.GlobalAveragePooling2D()(x))


def _features(extractor, paths, img_size, flip=False, batch_size=64):
    import tensorflow as tf
    from src.data.loader import _load_image
    ds = tf.data.Dataset.from_tensor_slices((paths, [0] * len(paths)))
    ds = ds.map(lambda p, l: _load_image(p, l, (img_size, img_size)),
                num_parallel_calls=tf.data.AUTOTUNE)
    if flip:
        ds = ds.map(lambda x, l: (tf.image.flip_left_right(x), l))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return extractor.predict(ds.map(lambda x, l: x), verbose=1)


def train_model_cached(splits, num_classes, epochs=40, patience=6, weights='imagenet',
                       img_size=224, cache_path=None, batch_size=256,
                       model_path='model/model_assets/trained_checkpoints/birds_classify.keras'):
    """Fast training for computers without a GPU.

    MobileNetV2 features are computed once per photo (plus a mirrored copy of each
    training photo as augmentation) and cached; only the softmax head is trained on
    them, which takes seconds per epoch. The head is then placed on a full model
    identical to build_model(), so prediction, evaluation and export are unchanged.
    No fine-tuning of the base: use train_model() on a GPU for that.
    """
    if weights is None:
        raise ValueError("--cached-features needs the ImageNet starting weights (drop --no-pretrained)")
    (tr_p, tr_y), (va_p, va_y) = splits.files['train'], splits.files['val']
    if cache_path and os.path.exists(cache_path):
        print(f"  Using cached features: {cache_path}")
        c = np.load(cache_path)
        x_tr, y_tr, x_va = c['x_tr'], c['y_tr'], c['x_va']
    else:
        extractor = _feature_extractor(img_size, weights)
        print(f"  Computing features: {len(tr_p)} training photos x 2 views, {len(va_p)} validation")
        x_tr = np.concatenate([_features(extractor, tr_p, img_size),
                               _features(extractor, tr_p, img_size, flip=True)])
        y_tr = np.concatenate([tr_y, tr_y])
        x_va = _features(extractor, va_p, img_size)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, x_tr=x_tr, y_tr=y_tr, x_va=x_va)
    y_va = np.array(va_y)

    # Balance classes: species with few photos count more, dampened by a square root.
    counts = np.bincount(y_tr, minlength=num_classes).astype(float)
    weights_per_class = np.sqrt(counts.max() / np.maximum(counts, 1))
    class_weight = {i: float(w / weights_per_class.mean()) for i, w in enumerate(weights_per_class)}

    feat_in = keras.Input(shape=(x_tr.shape[1],))
    h = layers.Dropout(0.3)(feat_in)
    head_out = layers.Dense(num_classes, activation='softmax', name='species')(h)
    head = keras.Model(feat_in, head_out)
    _compile(head, 1e-3)
    print(f"\n=== Training classifier head on cached features (up to {epochs} epochs) ===")
    history = head.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
        class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                                 restore_best_weights=True, verbose=1),
                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                     patience=3, verbose=1)])

    model, _ = build_model(num_classes, input_shape=(img_size, img_size, 3), weights=weights)
    model.layers[-1].set_weights(head.get_layer('species').get_weights())
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Saved {model_path}")
    return model, history


def predict_dataset(model, ds):
    """Run the model over a batched dataset; returns (y_true, probabilities)."""
    y_true, probs = [], []
    for images, labels in ds:
        probs.append(model.predict(images, verbose=0))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(probs)
