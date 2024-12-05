# ReactFace: Generating Multiple Appropriate Facial Reactions in Real-time Conversations

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](xxxxxxxx)
[![Paper1](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2305.15748)
[![Paper2](https://img.shields.io/badge/Paper-IEEE-green)](https://ieeexplore.ieee.org/abstract/document/10756784)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/lingjivoo/ReactFace)

</div>

## 📢 News

- Our paper has been accepted by IEEE Transactions on Visualization and Computer Graphics (TVCG)! 🎉🎉 (Oct/2024)

## 📋 Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.8+

### Setup Environment

#### Create and activate conda environment
```bash
conda create -n react python=3.9
conda activate react
```

#### Install PyTorch
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### Install PyTorch3D
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
```

#### Install other dependencies
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Data Preparation
<details>
<summary><b>Download and Setup Dataset</b></summary>

The REACT 2023/2024 Multimodal Challenge Dataset is compiled from the following public datasets for studying dyadic interactions:
- [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780)
- [RECOLA](https://ieeexplore.ieee.org/document/6553805)

Apply for data access at:
- [REACT 2023 Homepage](https://sites.google.com/cam.ac.uk/react2023/home)
- [REACT 2024 Homepage](https://sites.google.com/cam.ac.uk/react2024)

**Data organization (`data/`) follows this structure:**
```
data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files
```

Example data structure:
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.png
                       ├── ....
                       ├── 751.png
                   ├── ....
           ├── ....
       ├── RECOLA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
```

Important details:
- Task: Predict one role's reaction ('Expert' or 'Novice', 'P25' or 'P26') to the other
- 3D_FV_files contain 3DMM coefficients (expression: 52 dim, angle: 3 dim, translation: 3 dim)
- Video specifications:
  - Frame rate: 25 fps
  - Resolution: 256x256
  - Clip length: 751 frames (~30s)
  - Audio sampling rate: 44100
- CSV files for training/validation are available at: 'data/train.csv', 'data/val.csv', 'data/test.csv'

</details>

### 2. External Tool Preparation
<details>
<summary><b>Required Models and Tools</b></summary>

We use 3DMM coefficients for 3D listener/speaker representation and 3D-to-2D frame rendering.

1. **3DMM Model Setup**
   - Download [FaceVerse version 2 model](https://github.com/LizhenWangT/FaceVerse)
   - Place in `external/FaceVerse/data/`
   - Get pre-extracted data:
     - [3DMM coefficients](https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu)
     - [Reference files](https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4) (mean_face, std_face, reference_full)
     - Place in `external/FaceVerse/`

2. **PIRender Setup**
   - We use [PIRender](https://github.com/RenYurui/PIRender) for 3D-to-2D rendering
   - Download our retrained [checkpoint](https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA)
   - Place in `external/PIRender/`

</details>

### 3. Training
<details>
<summary><b>Training Options</b></summary>

Training with rendering during training:
```bash
python train.py \
  --batch-size 8 \
  --window-size 64 \
  --momentum 0.1 \
  --gpu-ids 0 \
  -lr 0.00002 \
  -e 200 \
  -j 4 \
  --sm-p 10 \
  --kl-p 0.00001 \
  --div-p 100 \
  --rendering \
  --outdir results/train-reactface
```

Training without rendering during validation (faster):
```bash
python train.py \
  --batch-size 8 \
  --window-size 64 \
  --momentum 0.1 \
  --gpu-ids 0 \
  -lr 0.00002 \
  -e 200 \
  -j 4 \
  --sm-p 10 \
  --kl-p 0.00001 \
  --div-p 100 \
  --outdir results/train-reactface
```

</details>

### 4. Evaluation
<details>
<summary><b>Evaluation Process</b></summary>

Run the following command to evaluate trained ReactFace:
```bash
python evaluate.py \
  --split test \
  --batch-size 16 \
  --window-size 8 \
  --momentum 0.9 \
  --gpu-ids 0 \
  -j 4 \
  --outdir results/eval \
  --resume results/training-reactface/best_checkpoint.pth
```

</details>

## Citation

If this work helps in your research, please cite the following papers:

```bibtex
@article{luo2024reactface,
    title={ReactFace: Online Multiple Appropriate Facial Reaction Generation in Dyadic Interactions},
    author={Luo, Cheng and Song, Siyang and Xie, Weicheng and Spitale, Micol and Ge, Zongyuan and Shen, Linlin and Gunes, Hatice},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    year={2024},
    publisher={IEEE}
}

@article{luo2023reactface,
    title={Reactface: Multiple appropriate facial reaction generation in dyadic interactions},
    author={Luo, Cheng and Song, Siyang and Xie, Weicheng and Spitale, Micol and Shen, Linlin and Gunes, Hatice},
    journal={arXiv preprint arXiv:2305.15748},
    year={2023}
}
```

## Acknowledgements

Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)
