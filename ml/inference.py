import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Agriculture",
    "Forest",
    "Industrial",
    "Urban",
    "Water"
]

VIT_URL = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"


# ─────────────────────────────────────────────────────────────────────────────
# ViT wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ViTWrapper(tf.keras.layers.Layer):

    def __init__(self, module, trainable_backbone=False, **kwargs):
        super().__init__(**kwargs)

        self._vit = module
        self.trainable = trainable_backbone

    def call(self, inputs, training=False):

        inputs = tf.cast(inputs, tf.float32)

        return self._vit(inputs)


# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path):

    print("Loading ViT-B/16 backbone...")

    vit_module = hub.load(VIT_URL)

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        dtype="float32"
    )

    features = ViTWrapper(
        vit_module,
        trainable_backbone=False,
        name="vit_backbone"
    )(inputs)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        name="head_dense"
    )(features)

    x = tf.keras.layers.Dropout(
        0.3,
        name="head_dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        5,
        activation="softmax",
        dtype="float32",
        name="predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    print("Loading trained weights...")

    model.load_weights(weights_path)

    print("✓ Model loaded successfully")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, image):

    processed_image = preprocess_image(image)

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }