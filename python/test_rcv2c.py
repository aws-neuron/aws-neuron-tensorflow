import tensorflow as tf
import tensorflow.neuron as tfn

'''
input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
inputs = [input0]
outputs = [dense0]
model = tf.keras.Model(inputs=inputs, outputs=outputs)
input0_tensor = tf.random.uniform([1, 3])
model_neuron = tfn.trace(model, input0_tensor)
'''

rn_50 = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

inputs = tf.random.uniform((1, 224,224,3))

model = tfn.trace(rn_50, inputs)
