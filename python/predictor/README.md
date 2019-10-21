### NeuronPredictor

The NeuronPredictor API provides simple interface to perform repeated inference on a pre-trained model.

```
predictor = NeuronPredictor(model_dir,
                             signature_def_key=None,
                             signature_def=None,
                             input_names=None,
                             output_names=None,
                             tags=None,
                             graph=None,
                             config=None,
                             use_ei=True)


output_dict = predictor(feed_dict)
```

The usage of NeuronPredictor is similar to TF predictor for saved model (https://www.tensorflow.org/api_docs/python/tf/contrib/predictor/from_saved_model)

_NeuronPredictor can be used in the following ways:_

```
//NeuronPredictor class picks inputs and outputs  from default serving signature def  with tag "serve". (similar to TF predictor)
predictor = NeuronPredictor(model_dir)

//NeuronPredictor class picks inputs and outputs from the using signature def picked using signtaure_def_key (similar to TF predictor)
predictor = NeuronPredictor(model_dir, signature_def_key='predict')

// Signature_def can be provided directly (similar to TF predictor)
predictor = NeuronPredictor(model_dir, signature_def= sig_def)

// User provides the input_names and output_names dict.
// if signature_def/signature_def_key is provided , those will be ignored.
// similar to TF predictor
predictor = NeuronPredictor(model_dir,
                             input_names,
                             output_names)

// tag is used to get the correct signature def. (similar to TF predictor)
predictor = NeuronPredictor(model_dir, tags='serve')
```

_Additional functionality in Neuron Predictor:_

- Supports frozen models.

```
// For Frozen graphs, model_dir takes a file name , input_names and output_names
// input_names and output_names should be provided in this case.
predictor = NeuronPredictor(model_dir,
                             input_names=None,
                             output_names=None )
```

- User can also disable usage of Neuron by using **use_neuron** flag which is defaulted to **True**.
  This is useful to test NeuronPredictor against Tensorflow Predictor.
- NeuronPredictor can also be created from TensorFlow Estimator. Given a trained Estimator, we first export a SavedModel:

  ```
  saved_model_dir = estimator.export_savedmodel(my_export_dir, serving_input_fn)
  Refer https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/predictor#predictor-from-a-savedmodel for more details
  ```

  NeuronPredictor can be constructed as follows:

  ```
  predictor = NeuronPredictor(export_dir=saved_model_dir)

  // Once the NeuronPredictor is created, inference is done using the following:
  output_dict = predictor(feed_dict)
  ```

#### Limitations:

