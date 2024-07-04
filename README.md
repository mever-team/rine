# Paper
This repository contains the implementation code for the [ECCV 2024](https://eccv2024.ecva.net/) accepted paper:

**Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection** (available at [arXiv:2402.19091](https://arxiv.org/abs/2402.19091))

**[<u>Christos Koutlis</u>](https://orcid.org/0000-0003-3682-408X), [<u>Symeon Papadopoulos</u>](https://orcid.org/0000-0002-5441-7341)**

![](https://github.com/mever-team/rine/blob/main/results/figs/fig1.png)
***Figure 1**. The RINE architecture. A batch of $`b`$ images is processed by CLIP's image encoder. The concatenation of the $`n`$ $`d`$-dimensional CLS tokens (one from each Transformer block) is first projected and then multiplied with the blocks' scores, estimated by the Trainable Importance Estimator (TIE) module. Summation across the second dimension results in one feature vector per image. Finally, after the second projection and the consequent classification head modules, two loss functions are computed. Binary cross-entropy $`\mathfrak{L}_{CE}`$ directly optimizes SID, while the contrastive loss $`\mathfrak{L}_{Cont.}`$ assists the training by forming a dense feature vector cluster per class.*

# News
:tada: **4/7/2024** Paper acceptance at [ECCV 2024](https://eccv2024.ecva.net/)

:sparkles: **29/2/2024** Pre-print release --> [arXiv:2402.19091](https://arxiv.org/abs/2402.19091)

:boom: **29/2/2024** Code and checkpoints release

# Setup
Clone the repository:
```
git clone https://github.com/mever-team/rine
```
Create the environment:
```
conda create -n rine python=3.9
conda activate rine
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Store the datasets in `data/`:
* Download the `ProGAN` training & validation, and the `GAN-based`, `Deepfake`, `Low-level-vision`, and `Perceptual loss` test sets as desrcibed in https://github.com/PeterWang512/CNNDetection
* Download the `Diffusion` test data as desrcibed in https://github.com/Yuheng-Li/UniversalFakeDetect
* Download the ``Latent Diffusion Training data`` as described in https://github.com/grip-unina/DMimageDetection
* Download the ``Synthbuster`` dataset as desrcibed in https://zenodo.org/records/10066460
* Download the ``MSCOCO`` dataset https://cocodataset.org/#home

The `data/` directory should look like:
```
data
└── coco
└── latent_diffusion_trainingset
└── RAISEpng
└── synthbuster
└── train
      ├── airplane	
      │── bicycle
      |     .
└── val
      ├── airplane	
      │── bicycle
      |     .
└── test					
      ├── progan	
      │── cyclegan   	
      │── biggan
      │      .
      │── diffusion_datasets
                │── guided
                │── ldm_200
                |       .
```

# Evaluation
To evaluate the 1-class, 2-class, and 4-class chechpoints as well as the LDM-trained model provided in `ckpt/` run `python scripts/validation.py`. The results will be displayed in terminal.

To get all the reported results (figures, tables) of the paper run `python scripts/results.py`.

# Re-run experiments
To reproduce the conducted experiments, re-run in the following order:
1. the 1-epoch hyperparameter grid experiments with `python scripts/experiments.py`
2. the ablation study with `python scripts/ablations.py`
3. the training duration experiments with `python scripts/epochs.py`
4. the training set size experiments with `python scripts/dataset_size.py`
5. the perturbation experiments with `python scripts/perturbations.py`
6. the LDM training experiments with `python scripts/diffusion.py`

Finally, to save the best 1-class, 2-class, and 4-class models (already stored in `ckpt/`) run `python scripts/best.py`, that re-trains the best configurations and stores the corresponding trainable model parts.

With this code snippet the whole project can be reproduced:
```
import subprocess

subprocess.run("python scripts/experiments.py", shell=True)
subprocess.run("python scripts/ablations.py", shell=True)
subprocess.run("python scripts/epochs.py", shell=True)
subprocess.run("python scripts/dataset_size.py", shell=True)
subprocess.run("python scripts/perturbations.py", shell=True)
subprocess.run("python scripts/diffusion.py", shell=True)
subprocess.run("python scripts/best.py", shell=True)
subprocess.run("python scripts/validation.py", shell=True)
subprocess.run("python scripts/results.py", shell=True)
```

# Demo
In `demo/`, we also provide code for inference on one real and one fake image from the DALL-E generative model. To demonstrate run `python demo/demo.py`.

# Citation
```
@article{koutlis2024leveraging,
  title={Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection},
  author={Koutlis, Christos and Papadopoulos, Symeon},
  journal={arXiv preprint arXiv:2402.19091},
  year={2024}
}
```

# Contact
Christos Koutlis (ckoutlis@iti.gr)