import tensorflow as tf
import tensorflow.neuron as tfn

class RemoveConstantsWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._weights = []
        for layer in self.model.layers:
            for weight in layer.weights:
                self._weights.append(weight)

    def call(self, inputs):
        return model(inputs, *self._weights)

input0 = tf.keras.layers.Input(3)
dense0 = tf.keras.layers.Dense(3)(input0)
inputs = [input0]
outputs = [dense0]
model = tf.keras.Model(inputs=inputs, outputs=outputs)
wrapped = RemoveConstantsWrapper(model)
input0_tensor = tf.random.uniform([1, 3])
model_neuron = tfn.trace(wrapped, input0_tensor)

'''
weights = []
for layer in model.layers:
    for weight in layer.weights:
        weights.append(weight)

rn_50 = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

inputs = tf.random.uniform((1, 224,224,3))
model = tfn.trace(rn_50, inputs)
#wrapped = RemoveConstantsWrapper(model)

'''

print(model_neuron(inputs,))
