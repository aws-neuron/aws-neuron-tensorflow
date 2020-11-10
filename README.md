# Introduction
This repository contains the source code of the AWS Neuron TensorFlow integration project.
It primarily serves the purpose of demonstrating how to integrate AWS Neuron into an existing,
self-maintained fork of TensorFlow. For detailed usage and examples on integrating your
TensorFlow based deep learning applications with AWS Neuron, please refer to the
[AWS Neuron SDK main documentation site](https://github.com/aws/aws-neuron-sdk/tree/master/docs/tensorflow-neuron).

# Build AWS Neuron TensorFlow integration
Here are the steps for building AWS Neuron TensorFlow integration.
It is available in the following two forms.
1. `tensorflow-neuron` pip whl, a Python package that adds on top of `tensorflow~=1.15.0`.
1. `tensorflow_model_server_neuron` binary executable, a special build of
[TensorFlow Serving](https://github.com/tensorflow/serving) (tf-serving) 1.15.0.

The AWS Neuron runtime is integrated into TensorFlow as a TensorFlow custom operator,
namely [NeuronOp](https://github.com/aws/aws-neuron-tensorflow/blob/main/runtime/ops/neuron_op.cc),
without any modification to the core TensorFlow runtime code. As a result, AWS customers may
bring in their own fork of TensorFlow 1.15, potentially with their own modifications,
and expect it to work with AWS Neuron seemlessly together.

Conceptually, `NeuronOp` is a [gRPC](https://grpc.io/) client based on the
[AWS Neuron Runtime Protocol Buffer](https://github.com/aws/aws-neuron-runtime-proto) interface.

## Install build tool
We recommend [Bazelisk](https://github.com/bazelbuild/bazelisk) (go-version)
which is "a user-friendly launcher for [Bazel](https://bazel.build/)".
1. Install [go](https://golang.org/)
    - On Debian-based OS (e. g., Ubuntu): `sudo apt-get install golang`
    - On AmazonLinux2: `sudo yum install golang`
    - On other CentOS-based OS:
        1. `sudo rpm --import https://mirror.go-repo.io/centos/RPM-GPG-KEY-GO-REPO`
        1. `curl -s https://mirror.go-repo.io/centos/go-repo.repo | sudo tee /etc/yum.repos.d/go-repo.repo`
        1. `sudo yum install golang`
    - Verify by running `go version`
1. Install Bazelisk (from [https://github.com/bazelbuild/bazelisk#requirements]) and name it as `bazel`
    1. `go get github.com/bazelbuild/bazelisk`
    1. `mkdir -p $HOME/bin`
    1. `install $(go env GOPATH)/bin/bazelisk $HOME/bin/bazel`
    1. Verify by running `bazel version`

## Build procedures
### `tensorflow-neuron` pip whl
1. Install Python3 developer package
    - On Debian-based OS (e. g., Ubuntu): `sudo apt install python3-dev python3-pip`
    - On AmazonLinux2 or other CentOS-based OS: `sudo yum install python3-devel python3-pip`
1. Setup build `venv` and install dependencies
    1. `python3 -m venv env_tfn`
    1. `source activate ./env_tfn/bin/activate`
    1. `pip install pip numpy wheel`
    1. `pip install keras_preprocessing --no-deps`
1. Clone `tensorflow` source code and setup `tensorflow-neuron` and Neuron runtime `proto` directories
    1. `git clone https://github.com/tensorflow/tensorflow.git -b v1.15.4 tensorflow`
    1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow/tensorflow/neuron`
    1. `git clone https://github.com/aws/aws-neuron-runtime-proto ./tensorflow/tensorflow/neuron/runtime/proto`
1. Build `tensorflow-neuron` pip whl
    1. `cd tensorflow`
    1. `USE_BAZEL_VERSION=0.26.1 ./configure`
    1. `USE_BAZEL_VERSION=0.26.1 bazel build --incompatible_remap_main_repo --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/neuron:build_pip_package`
    1. `./bazel-bin/tensorflow/neuron/build_pip_package ./`
    1. pip whl can be found by `ls tensorflow_neuron-*.whl`
1. (Optional) Validate the `tensorflow-neuron` pip whl
    1. `mkdir ../rundir`
    1. `cd ../rundir`
    1. `pip install pytest neuron-cc tensorflow/tensorflow_neuron-*.whl --extra-index-url=https://pip.repos.neuron.amazonaws.com`
    1. `env NEURON_TF_COMPILE_ONLY=1 pytest --pyargs tensorflow_neuron`, all tests should pass.
        - If tests are running on `inf1` instances with `aws-neuron-runtime` installed,
        then you may simply run `pytest --pyargs tensorflow_neuron` and expect all tests passing.

### `tensorflow_model_server_neuron` binary executable
We recommend building `tensorflow_model_server_neuron` in docker image
`tensorflow/serving:1.15.0-devel` which includes the source code of
tf-serving 1.15.0 and its entire build dependency environment.
1. `docker run -it --rm -v $(pwd):/host_workspace tensorflow/serving:1.15.0-devel bash`
    - This step should let you drop into `/tensorflow-serving` which has the same content as
    https://github.com/tensorflow/serving/tree/1.15.0.
1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow_serving/neuron`
1. `git clone https://github.com/aws/aws-neuron-runtime-proto ./tensorflow_serving/neuron/runtime/proto`
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
1. `cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /host_workspace/tensorflow_model_server_neuron`
1. Get out of Docker container by `exit`
1. Verify by running `./tensorflow_model_server_neuron --help`
