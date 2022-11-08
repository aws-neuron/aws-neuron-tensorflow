# Introduction
This repository contains the source code of the AWS Neuron TensorFlow integration project.
It primarily serves the purpose of demonstrating how to integrate AWS Neuron into an existing,
self-maintained fork of TensorFlow. For detailed usage and examples on integrating your
TensorFlow based deep learning applications with AWS Neuron, please refer to the
[AWS Neuron SDK main documentation site](https://github.com/aws/aws-neuron-sdk/tree/master/docs/tensorflow-neuron).

# Build AWS Neuron TensorFlow integration
Here are the steps for building AWS Neuron TensorFlow integration.
It is available in the following two forms.
1. `tensorflow-neuron` pip whl, a Python package that adds on top of `tensorflow`.
1. `tensorflow_model_server_neuron` binary executable, a special build of
[TensorFlow Serving](https://github.com/tensorflow/serving) (tf-serving).

The AWS Neuron runtime is integrated into TensorFlow as a TensorFlow custom operator,
namely [NeuronOp](https://github.com/aws/aws-neuron-tensorflow/blob/main/runtime/ops/neuron_op.cc),
without any modification to the core TensorFlow code. As a result, AWS customers may
bring in their own fork of TensorFlow, potentially with their own modifications,
and expect it to work with AWS Neuron seemlessly together.

The open source distribution of tensorflow-neuron requires deb/rpm package
`aws-neuronx-runtime-lib` at run-time. For more information, please refer to
[Introducing Packaging and installation changes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/neuron2.x/neuron230-packages-changes.html)
and
[AWS Neuron Runtime 2.x (libnrt.so)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/appnotes/neuron-components/introducing-libnrt.html#introduce-libnrt).

## Install build tool
We recommend [Bazelisk](https://github.com/bazelbuild/bazelisk)
which is "a user-friendly launcher for [Bazel](https://bazel.build/)".
1. Install Bazelisk (from [https://github.com/bazelbuild/bazelisk#installation]) and name it as `bazel`
    1. `mkdir -p $HOME/bin`
    1. `curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64 --output $HOME/bin/bazel`
    1. `chmod +x $HOME/bin/bazel`
    1. `export PATH=$PATH:$HOME/bin`
    1. Verify by running `bazel version`

## Build procedures
### `tensorflow-neuron` 2.x pip whl
1. Install Python3 developer package
    - On Debian-based OS (e. g., Ubuntu): `sudo apt install python3-dev python3-pip python3-venv`
    - On AmazonLinux2 or other CentOS-based OS: `sudo yum install python3-devel python3-pip`
1. Setup build `venv` and install dependencies
    1. `python3 -m venv env_tfn`
    1. `source ./env_tfn/bin/activate`
    1. `pip install pip -U`
    1. `pip install numpy==1.20.0 wheel six`
    1. `pip install keras_preprocessing --no-deps`
1. Clone `tensorflow` source code and setup `tensorflow-neuron` directory
    1. `git clone https://github.com/tensorflow/tensorflow.git -b v2.8.3 tensorflow`
    1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow/tensorflow/neuron`
1. Build `tensorflow-neuron` pip whl
    1. `cd tensorflow`
    1. `./configure`
    1. `bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/neuron:build_pip_package`
    1. `./bazel-bin/tensorflow/neuron/build_pip_package ./`
    1. pip whl can be found by `ls tensorflow_neuron-*.whl`
1. (Optional) Validate the `tensorflow-neuron` pip whl on an `inf1` instance with pre-installed `aws-neuronx-dkms` and `aws-neuronx-runtime-lib`
    1. Copy `tensorflow_neuron-*.whl` to the `inf1` instance's `$HOME` directory, e. g. `scp tensorflow_neuron-*.whl my_inf1:~/`
    1. On the `inf1` instance:
        1. `mkdir ~/rundir`
        1. `cd ~/rundir`
        1. `python3 -m venv env_tfn`
        1. `source ./env_tfn/bin/activate`
        1. `pip install pip -U`
        1. `pip install pytest`
        1. `pip install neuron-cc ~/tensorflow_neuron-*.whl --extra-index-url=https://pip.repos.neuron.amazonaws.com`
        1. `pytest --pyargs tensorflow_neuron -k 'custom or dot or batchnorm'`, all tests should pass.

### `tensorflow_model_server_neuron` 2.x binary executable
We recommend building and running `tensorflow_model_server_neuron` on docker image
`tensorflow/serving:2.8.3-devel` which includes the source code of
tf-serving 2.8.3 and its entire build dependency environment. To install docker, please refer to
https://docs.docker.com/engine/install/.
1. `docker run -it --rm -v $(pwd):/host_workspace tensorflow/serving:2.8.3-devel bash`
    - This step should let you drop into `/tensorflow-serving` which has the same content as
    https://github.com/tensorflow/serving/tree/2.8.3.
1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow_serving/neuron`
1. `sed -i 's/SUPPORTED_TENSORFLOW_OPS = /SUPPORTED_TENSORFLOW_OPS = ["\/\/tensorflow_serving\/neuron\/runtime:all_ops"] + /g' ./tensorflow_serving/model_servers/BUILD`
    - If the sed command fails to execute, you may choose to manually edit
    `tensorflow_serving/model_servers/BUILD` to let `SUPPORTED_TENSORFLOW_OPS` include
    Bazel target `"//tensorflow_serving/neuron/runtime:all_ops"`.
1. `bazel build //tensorflow_serving/model_servers:tensorflow_model_server`
1. `install bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server ./tensorflow_model_server_neuron`
1. Verify by installing `aws-neuronx-runtime-lib` and running `tensorflow_model_server_neuron`
    1. `echo "deb [trusted=yes] https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list`
    1. `apt-get update && apt-get install -y aws-neuronx-runtime-lib`
    1. `./tensorflow_model_server_neuron --help`

### `tensorflow-neuron` 1.x pip whl
1. Install Python3 developer package
    - On Debian-based OS (e. g., Ubuntu): `sudo apt install python3-dev python3-pip python3-venv`
    - On AmazonLinux2 or other CentOS-based OS: `sudo yum install python3-devel python3-pip`
1. Setup build `venv` and install dependencies
    1. `python3 -m venv env_tfn`
    1. `source ./env_tfn/bin/activate`
    1. `pip install pip -U`
    1. `pip install numpy==1.18.5 wheel six`
    1. `pip install keras_preprocessing --no-deps`
1. Clone `tensorflow` source code and setup `tensorflow-neuron` directory
    1. `git clone https://github.com/tensorflow/tensorflow.git -b v1.15.5 tensorflow`
    1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow/tensorflow/neuron`
1. Build `tensorflow-neuron` pip whl
    1. `cd tensorflow`
    1. `git checkout refs/tags/v1.15.5`
    1. `USE_BAZEL_VERSION=0.26.1 ./configure`
    1. `USE_BAZEL_VERSION=0.26.1 bazel build --incompatible_remap_main_repo --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/neuron:build_pip_package`
    1. `./bazel-bin/tensorflow/neuron/build_pip_package ./`
    1. pip whl can be found by `ls tensorflow_neuron-*.whl`
1. (Optional) Validate the `tensorflow-neuron` pip whl
    1. `mkdir ../rundir`
    1. `cd ../rundir`
    1. `pip install pytest`
    1. `pip install neuron-cc ../tensorflow/tensorflow_neuron-*.whl --extra-index-url=https://pip.repos.neuron.amazonaws.com`
    1. `env NEURON_TF_COMPILE_ONLY=1 pytest --pyargs tensorflow_neuron`, all tests should pass.
        - If tests are running on `inf1` instances with pre-installed `aws-neuronx-dkms` and `aws-neuronx-runtime-lib`,
        then you may simply run `pytest --pyargs tensorflow_neuron` and expect all tests passing.
        - Known issue: if you have `h5py>=3` installed, some Keras related tests may fail due to https://github.com/tensorflow/tensorflow/issues/44467

### `tensorflow_model_server_neuron` 1.x binary executable
We recommend building and running `tensorflow_model_server_neuron` on docker image
`tensorflow/serving:1.15.0-devel` which includes the source code of
tf-serving 1.15.0 and its entire build dependency environment. To install docker, please refer to
https://docs.docker.com/engine/install/.
1. `docker run -it --rm -v $(pwd):/host_workspace tensorflow/serving:1.15.0-devel bash`
    - This step should let you drop into `/tensorflow-serving` which has the same content as
    https://github.com/tensorflow/serving/tree/1.15.0.
1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow_serving/neuron`
1. `git apply ./tensorflow_serving/neuron/runtime/serving_neuron_op.diff`
    - All this patch does is to register `NeuronOp` into tf-serving by adding
    the following line of code into Bazel BUILD file `tensorflow_serving/model_servers/BUILD`.
    ```
    SUPPORTED_TENSORFLOW_OPS.append("//tensorflow_serving/neuron/runtime:all_ops")
    ```
    - If the patch fails to apply due to file content conflicts, you may choose to manually edit
    `tensorflow_serving/model_servers/BUILD` to let `SUPPORTED_TENSORFLOW_OPS` include
    Bazel target `"//tensorflow_serving/neuron/runtime:all_ops"`.
1. `bazel build //tensorflow_serving/model_servers:tensorflow_model_server`
1. `install bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server ./tensorflow_model_server_neuron`
1. Verify by installing `aws-neuronx-runtime-lib` and running `tensorflow_model_server_neuron`
    1. `echo "deb [trusted=yes] https://apt.repos.neuron.amazonaws.com bionic main" > /etc/apt/sources.list.d/neuron.list`
    1. `apt-get update && apt-get install -y aws-neuronx-runtime-lib`
    1. `./tensorflow_model_server_neuron --help`
