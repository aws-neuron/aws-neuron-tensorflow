#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

function usage() {
    echo "Usage:"
    echo "$0 dstdir [platform] [version_tag]"
    echo ""
    exit 1
}

function is_absolute {
    [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
    is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function main() {
    if [[ -z "$1" ]]; then
        echo "No destination dir provided"
        usage
        exit 1
    fi
    DSTDIR="$(real_path $1)"
    if [[ -z "$2" ]]; then
        PLATFORM=""
    else
        PLATFORM="-p $2"
        echo "Building pip wheel for platform ${2}"
    fi
    if [[ -z "$3" ]]; then
        VERSION="$(git describe --tags)"
        VERSION="${VERSION:1}.dev$(date +'%Y%m%d')"
        echo "Determined version tag ${VERSION} from tensorflow git repo"
    else
        VERSION="$3"
    fi

    if [ ! -d bazel-bin/tensorflow ]; then
        echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
        exit 1
    fi
    TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
    cp -R bazel-bin/tensorflow/neuron/build_pip_package.runfiles/org_tensorflow/tensorflow/neuron "${TMPDIR}/tensorflow_neuron"
    mkdir "${TMPDIR}/tensorflow-plugins"
    mv ${TMPDIR}/tensorflow_neuron/python/libaws_neuron_plugin.* "${TMPDIR}/tensorflow-plugins"
    mv ${TMPDIR}/tensorflow_neuron/tensorflow.py "${TMPDIR}/"

    # whether to put api definition and aws_neuron_tf2hlo under tensorflow_core vs tensorflow
    UNDER_TF_CORE=$(python3 -c "from distutils.version import LooseVersion; print(LooseVersion(\"${VERSION}\") < LooseVersion('2.2'))")
    if [[ ${UNDER_TF_CORE} == "True" ]]; then
        TF_CORE_PATH="${TMPDIR}/tensorflow_core"
        mkdir -p "${TMPDIR}/tensorflow/neuron/"
        cp "${TMPDIR}/tensorflow_neuron/api/__init__.py" "${TMPDIR}/tensorflow/neuron/"
    else
        TF_CORE_PATH="${TMPDIR}/tensorflow"
    fi
    mkdir -p "${TF_CORE_PATH}/neuron/"
    cp "${TMPDIR}/tensorflow_neuron/api/__init__.py" "${TF_CORE_PATH}/neuron/"
    mv "${TMPDIR}/tensorflow_neuron/tf2hlo/" "${TF_CORE_PATH}/neuron/"
    sed "s/_VERSION/${VERSION}/g" tensorflow/neuron/setup.py > "${TMPDIR}/setup.py"
    echo "__version__ = '${VERSION}'" >> "${TMPDIR}/tensorflow_neuron/__init__.py"

    # Before we leave the top-level directory, make sure we know how to
    # call python.
    if [[ -e tools/python_bin_path.sh ]]; then
        source tools/python_bin_path.sh
    fi

    echo $(date) : "=== Building wheel"
    cd "${TMPDIR}"
    "${PYTHON_BIN_PATH:-python}" setup.py -q bdist_wheel ${PLATFORM}
    mkdir -p "${DSTDIR}"
    cp "${TMPDIR}"/dist/* "${DSTDIR}"
    echo $(date) : "=== Output wheel file is in: ${DSTDIR}"

    rm -rf "${TMPDIR}"
}

main "$@"
