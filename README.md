## Install build tool
We recommend [`Bazelisk`](https://github.com/bazelbuild/bazelisk) (go-version) which is "a user-friendly launcher for [`Bazel`](https://bazel.build/)".
1. Install [`go`](https://golang.org/)
    - On Debian-based OS (e. g., Ubuntu): `sudo apt-get install golang`
    - On AmazonLinux2: `sudo yum install golang`
    - On other CentOS-based OS:
        1. `sudo rpm --import https://mirror.go-repo.io/centos/RPM-GPG-KEY-GO-REPO`
        1. `curl -s https://mirror.go-repo.io/centos/go-repo.repo | sudo tee /etc/yum.repos.d/go-repo.repo`
        1. `sudo yum install golang`
        1. Verify by running `go version`
2. Install `Bazelisk` (from [https://github.com/bazelbuild/bazelisk#requirements]) and alias it to `bazel`
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
1. Build `tensorflow-neuron`
    1. `cd tensorflow`
    1. `USE_BAZEL_VERSION=0.26.1 ./configure`
    1. `USE_BAZEL_VERSION=0.26.1 bazel build --incompatible_remap_main_repo --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/neuron:build_pip_package`

### `tensorflow_model_server_neuron` binary executable
We recommend building `tensorflow_model_server_neuron` in docker image `tensorflow/serving:1.15.0-devel` which include the source code of `tensorflow/serving` 1.15.0 and its build dependency environment.
1. `docker run -it --rm -v $(pwd):/host_workspace tensorflow/serving:1.15.0-devel bash`
    - This step should let you drop into `/tensorflow-serving` which has the same content as https://github.com/tensorflow/serving/tree/1.15.0
1. `git clone https://github.com/aws/aws-neuron-tensorflow ./tensorflow_serving/neuron`
1. `git clone https://github.com/aws/aws-neuron-runtime-proto ./tensorflow_serving/neuron/runtime/proto`
1. `git apply ./tensorflow_serving/neuron/runtime/serving_neuron_op.diff`
1. `bazel build //tensorflow_serving/model_servers:tensorflow_model_server`
1. `cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /host_workspace/tensorflow_model_server_neuron`
1. Get out of Docker container by `exit`
1. Verify by running `./tensorflow_model_server_neuron --help`
