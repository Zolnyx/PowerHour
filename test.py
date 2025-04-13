import tensorflow as tf

# Load the model as SavedModel format
model_path = "working_model_1"
try:
    model = tf.saved_model.load(model_path)
    print("✅ Model loaded as a SavedModel. It might need conversion.")
except Exception as e:
    print(f"❌ Failed to load model as a SavedModel: {e}")

# Load as a Keras H5 file
try:
    model = tf.keras.models.load_model(model_path + ".h5")
    print("✅ Model loaded as a Keras H5 file.")
except Exception as e:
    print(f"❌ Failed to load model as a H5 file: {e}")
