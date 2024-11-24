#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# should match env name from YAML
ENV_NAME=afford

pushd "${ROOT_DIR}"

    # setup conda
    CONDA_DIR="$(conda info --base)"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

    # deactivate the env, if it is active
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
        conda deactivate
    fi

    # !!! this removes existing version of the env
    conda remove -y -n "${ENV_NAME}" --all

    # create the env from YAML
    conda env create -f ./rlgpu_conda_env.yml
    if [ $? -ne 0 ]; then
        echo "*** Failed to create env"
        exit 1
    fi

    # activate env
    conda activate "${ENV_NAME}"
    if [ $? -ne 0 ]; then
        echo "*** Failed to activate env"
        exit 1
    fi

    # double check that the correct env is active
    ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
    if [ "${ENV_NAME}" != "${ACTIVE_ENV_NAME}" ]; then
        echo "*** Env is not active, aborting"
        exit 1
    fi

    conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 cudatoolkit=10.1 -c pytorch -c nvidia
    conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -y -c  bottler nvidiacub

    conda install -y pytorch3d -c pytorch3d

    # for install requirements
    pip install -r requirements.txt

    # install isaacgym package
    pip install -e .

popd

# install vgn package
pip install trimesh
pip install urdfpy
pip install catkin_pkg
pip install -e ./src

# installing torch-scatter based on torch 1.7.0 and cuda 10.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64
# pip -v install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
conda install -y pytorch-scatter -c pyg

pip install -U scikit-learn
pip install seaborn
pip install umap-learn
pip install moviepy
pip install pptk
pip install opencv-python
pip install scikit-image
pip install timeout-decorator
# pip install pytorch3d
# pip install pytorch-ignite
pip install torchgeometry

# for new scripts
pip install einops
pip install transformers
pip install ftfy
pip install constants
# pip install graspnetAPI
pip install wandb

echo "SUCCESS"
