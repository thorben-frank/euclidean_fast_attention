![workflow-test-ci](https://github.com/thorben-frank/euclidean_fast_attention/actions/workflows/CI.yml/badge.svg)
[![preprint-link](https://img.shields.io/badge/paper-arxiv.org-B31B1B)](https://arxiv.org/abs/2412.08541)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14750286.svg)](https://doi.org/10.5281/zenodo.14750286)

![Logo](overview.png)

### Euclidean Fast Attention
Reference implementation of the Euclidean fast attention (EFA) algorithm, presented in the paper 
[*Euclidean Fast Attention: Machine Learning Global Atomic Representations at Linear Cost*](https://arxiv.org/abs/2412.08541).
#### Installation
The code in this repository can be either used with CPU or with GPU. If you want to use GPU, you have to install the 
corresponding JAX installation via 
```shell script
# On GPU
pip install --upgrade pip
pip install "jax[cuda12]"
```
If you want to run the code on CPU, e.g. for testing on your local machine which does not have a GPU, you can do
```shell script
# On CPU
pip install --upgrade pip
pip install jax
```
Note, that the code will run much fast on GPU than on CPU, so training is ideally performed on a GPU. More 
details about JAX installation can be found [here](https://jax.readthedocs.io/en/latest/installation.html).

Afterwards, you clone the EFA repository and install via
```shell script
# Euclidean fast attention installation
git clone https://github.com/thorben-frank/euclidean_fast_attention.git
cd euclidean_fast_attention
pip install .
```

#### Examples
For example usages check the `examples/` folder. It contains an examples for basic usage of the `EuclideanFastAttention` 
`flax` module. There are notebooks to train the models for the NaCl cluster systems and the EFA energy predictor and also a notebook 
which shows how to obtain energy scans from trained models.

### Datasets
A few data sets that are used throughout the examples are included here in the repository under the `datasets` folder.
All datasets can be found in the corresponding [zenodo repository](https://doi.org/10.5281/zenodo.14750286). Download the data, unpack and move the `datasets` folder to `${DATA_PATH}`.
The structure should then be `${DATA_PATH}/datasets/...`.

### Training and Evaluation
We provide a script for training on SN2, dimers, cumulene, MD17, MD22, 3BPA, the BIGDML materials data set and the 4GHDNNP benchmark. 
Here we show how to train the Au2MgO from the 4GHDNNP dataset. First create a folder where to save the training checkpoints
```shell script

mkdir ~/git/euclidean_fast_attention/runs

python ~/git/euclidean_fast_attention/euclidean_fast_attention/main.py \
    --config ~/git/euclidean_fast_attention/euclidean_fast_attention/configs/config.py \
    --config.wandb.group 4ghdnnp \
    --config.wandb.name efa_aumgo \
    --optimizer_config ~/git/euclidean_fast_attention/euclidean_fast_attention/configs/optimizer/default.py \
    --model_config ~/git/euclidean_fast_attention/euclidean_fast_attention/configs/model/4ghdnnp_base_model_pbc.py \
    --trainer_config ~/git/euclidean_fast_attention/euclidean_fast_attention/configs/trainer/4ghdnnp_AuMgO.py \
    --trainer_config.datafile "${DATA_PATH}/datasets/4ghdnnp/AuMgO_preprocessed.npz" \
    --workdir ~/git/euclidean_fast_attention/runs/4gdhnnp/aumgo/efa

```
Parameters can either be changed in the config directly, or via the command line. I.e., adding `--model_config.era_use_in_iterations=""` will result 
in a local model that does not use EFA. You can find the configs files used for the other data sets and models at `git/euclidean_fast_attention/configs`.
In the configs, there is the option `auto_eval` which if enabled calculates the metrics on the remaining test set data when training is finished.

#### Citation
If you find this repository useful or use the Euclidean fast attention algorithm in your research please
consider citing the corresponding paper
```
@article{frank2024euclidean,
  title={Euclidean Fast Attention: Machine Learning Global Atomic Representations at Linear Cost},
  author={Frank, J Thorben and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Unke, Oliver T},
  journal={arXiv preprint arXiv:2412.08541},
  year={2024}
}
```


