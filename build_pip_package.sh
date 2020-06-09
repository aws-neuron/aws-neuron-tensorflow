#!/usr/bin/env bash

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
    TFPYDIR="${TMPDIR}/tensorflow_core"
    mkdir -p "${TFPYDIR}"
    cp -R bazel-bin/tensorflow/neuron/build_pip_package.runfiles/org_tensorflow/tensorflow/neuron "${TFPYDIR}/"
    TFPLGNDIR="${TMPDIR}/tensorflow-plugins"
    mkdir "${TFPLGNDIR}"
    mv "${TFPYDIR}/neuron/convert/_aws_neuron_whitelist_partition_swig.so" "${TFPLGNDIR}/"
    sed "s/_VERSION/${VERSION}/g" tensorflow/neuron/setup.py > "${TMPDIR}/setup.py"

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
