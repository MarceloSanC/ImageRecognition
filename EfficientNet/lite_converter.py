#Code to convert h5 to tflite
import tensorflow as tf

model =tf.keras.models.load_model("EfficientNetV2S_8")

#Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Implement optimization strategy for smaller model sizes
#converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open("EfficientNetV2S_8.tflite", "wb").write(tflite_model)


