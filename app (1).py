import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
NUM_CLASSES = 29
# =========================
# Page config
# =========================
st.set_page_config(page_title="ASL Recognition", layout="centered")
st.title("🤟 ASL Alphabet Recognition")

# =========================
# Load class names
# =========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Keep classifier size consistent with class_names.json
NUM_CLASSES = len(class_names)

# =========================
# Model loader
# =========================
@st.cache_resource
def load_selected_model(model_name):
    if model_name == "InceptionV3":
        return load_model("models/inception.h5")
    elif model_name == "ResNet50":
        return load_model("models/resnet.h5")
    else:
        # Prefer loading the full saved model to avoid layer mismatches
        try:
            return load_model("models/efficientnet.h5", compile=False)
        except Exception:
            # Fallback: rebuild architecture and load weights by name, skipping mismatches
            base = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3),
                classifier_activation='softmax'
            )

            x = GlobalAveragePooling2D()(base.output)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.3)(x)
            out = Dense(NUM_CLASSES, activation="softmax")(x)

            model = Model(base.input, out)
            try:
                model.load_weights("models/efficientnet.h5")
            except Exception:
                model.load_weights("models/efficientnet.h5", by_name=True, skip_mismatch=True)
            return model

# =========================
# Image preprocessing
# =========================
def preprocess_image(pil_image, model_name):
    img = np.array(pil_image)
    img = cv2.resize(img, (224, 224))

    if model_name == "EfficientNet":
        # EfficientNet expects RGB inputs normalized to [-1, 1]
        img = eff_pre(img.astype("float32"))
    else:
        # Default: scale to [0, 1]
        img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)
    return img

# =========================
# Grad-CAM
# =========================
def _iter_layers_recursive(layer):
    yield layer
    if isinstance(layer, tf.keras.Model):
        for sub in layer.layers:
            yield from _iter_layers_recursive(sub)


def _find_layer_by_name(model, layer_name: str):
    for lyr in _iter_layers_recursive(model):
        if getattr(lyr, "name", None) == layer_name:
            return lyr
    return None


def _find_last_conv_layer(model):
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.Conv2DTranspose,
    )
    layers = list(_iter_layers_recursive(model))
    for lyr in reversed(layers):
        if isinstance(lyr, conv_types):
            return lyr
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Try the provided layer name; if missing (common with nested SavedModels), fall back.
    preferred = _find_layer_by_name(model, last_conv_layer_name) if last_conv_layer_name else None
    last_conv_layer = preferred or _find_last_conv_layer(model)
    if last_conv_layer is None:
        raise ValueError("Could not find a Conv2D-like layer for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Some SavedModels return multiple outputs; use the first by default.
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = tf.convert_to_tensor(predictions)
        # Ensure shape is (batch, classes)
        if predictions.shape.rank == 1:
            predictions = predictions[tf.newaxis, :]
        # Required: otherwise gradients can be None for intermediate tensors
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # pred_index may be a Tensor; ensure scalar int32 for indexing
        pred_index = tf.cast(pred_index, tf.int32)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError(
            f"Grad-CAM failed (gradients are None). Layer used: {last_conv_layer.name}."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = tf.where(denom > 0, heatmap / denom, tf.zeros_like(heatmap))
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # original_img is RGB (from PIL/numpy). OpenCV blending expects BGR for consistency.
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    blended_bgr = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)



model_choice = st.selectbox(
    "Choose Model",
    ["InceptionV3", "ResNet50", "EfficientNet"]
)

uploaded_file = st.file_uploader(
    "Upload an ASL hand image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_selected_model(model_choice)
    
        
    # Preprocess (match model-specific normalization)
    img = preprocess_image(image, model_choice)

    # Predict
    raw_preds = model.predict(img)
    if isinstance(raw_preds, (list, tuple)):
        raw_preds = raw_preds[0]
    raw_preds_arr = np.asarray(raw_preds)
    preds = raw_preds_arr.squeeze()
    if preds.ndim != 1:
        st.error(
            "This model is not outputting class probabilities. "
            f"Got prediction output shape {raw_preds_arr.shape}. "
            f"Expected (1, {len(class_names)}) for a {len(class_names)}-class softmax classifier. "
            "Your models/efficientnet.h5 is likely a feature extractor (include_top=False) rather than the trained 29-class model."
        )
        st.stop()

    if len(preds) != len(class_names):
        st.error(
            f"Model outputs {len(preds)} classes, but class_names.json has {len(class_names)} labels. "
            "This usually means you loaded a different model (e.g., ImageNet 1000-class) or mismatched weights."
        )
        st.stop()

    top3_idx = [int(i) for i in np.argsort(preds)[-3:][::-1]]
    
    st.subheader("Top 3 Predictions")
   
    
    for i, idx in enumerate(top3_idx, start=1):
        st.write(f"{i}. **{class_names[idx]}** — {preds[idx]*100:.2f}%")
    st.subheader("Grad-CAM Visualization")

    # Last conv layer names (adjust if needed)
    last_conv_layer_map = {
        "InceptionV3": "mixed10",
        "ResNet50": "conv5_block3_out",
        "EfficientNet": "top_conv"
        }

    last_conv_layer = last_conv_layer_map[model_choice]

    heatmap = make_gradcam_heatmap(
        img,
        model,
        last_conv_layer,
        pred_index=top3_idx[0]
    )

    original_img = np.array(image.resize((224, 224)))
    cam_image = overlay_gradcam(original_img, heatmap)

    st.image(cam_image, caption="Grad-CAM", use_column_width=True)