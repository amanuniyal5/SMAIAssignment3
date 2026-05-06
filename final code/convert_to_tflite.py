import onnx
import onnx2tf
import tensorflow as tf
import shutil

onnx_model_path = "model.onnx"
static_onnx_path = "model_static.onnx"
tflite_model_path = "model.tflite"
tf_model_path = "saved_model"

print("Fixing dynamic batch size natively in ONNX...")
# 1. Load the original ONNX model
onnx_model = onnx.load(onnx_model_path)

# 2. Force the dynamic 'batch' dimension to be exactly 1
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

# 3. Save the fixed model
onnx.save(onnx_model, static_onnx_path)
print("Static ONNX model saved!")

print("Starting ONNX to TensorFlow conversion...")
# Convert the NEW static model (Removed batch_size=1 to avoid the download bug)
onnx2tf.convert(
    input_onnx_file_path=static_onnx_path,
    output_folder_path=tf_model_path,
    output_signaturedefs=True,
)

print(f"TensorFlow model saved to {tf_model_path}")

print("Starting TensorFlow to TFLite conversion...")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
print(f"Saving TFLite model to {tflite_model_path}...")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

# Clean up the intermediate directory
shutil.rmtree(tf_model_path)

print("Conversion complete! The model is now perfectly static.")