# ReactFace: Multiple Appropriate Facial Reaction Generation in Dyadic Interactions

## Under construction. You can also refer to our baseline in [REACT 2023 Multimodal Challenge](https://github.com/reactmultimodalchallenge/baseline_react2023) first.

[[Homepage]](xxxxxxxx)  [[Reference Paper]](https://arxiv.org/pdf/2305.15748) [[Code]](https://github.com/lingjivoo/ReactFace)


📢 News
=
Our paper is accepted by IEEE Transactions on Visualization and Computer Graphics (TVCG)


## 🛠️ Installation

### Basic requirements

- Python 3.8+ 
- PyTorch 1.9+
- CUDA 11.8+

### Install Python dependencies (all included in 'requirements.txt')

```shell
conda create -n react python=3.9
conda activate react
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install -r requirements.txt
```

## 👨‍🏫 Get Started 
<details><summary> <b> Data Download</b> </summary>
<p>

- The REACT 2023 Multimodal Challenge Dataset is a compilation of recordings from the following three publicly available datasets for studying dyadic interactions: [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780), and [RECOLA](https://ieeexplore.ieee.org/document/6553805).

- Participants can apply for the data at our [Homepage](https://sites.google.com/cam.ac.uk/react2023/home).


**Data organization (`data/`) is listed below:**
```data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files```
The example of data structure.
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
       ├── UDIVA
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
       ├── UDIVA
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
       ├── UDIVA
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
       ├── UDIVA
            
```
 
- The task is to predict one role's reaction ('Expert' or 'Novice',  'P25' or 'P26'....) to the other ('Novice' or 'Expert',  'P26' or 'P25'....).
- 3D_FV_files involve extracted 3DMM coefficients (including expression (52 dim), angle (3 dim) and translation (3 dim) coefficients.
- The frame rate of processed videos in each site is 25 (fps = 25), height = 256, width = 256. And each video clip has 751 frames (about 30s), The samping rate of audio files is 44100. 
- The csv files for baseline training and validation dataloader are now avaliable at 'data/train.csv' and 'data/val.csv'
 
 
</p>
</details>



<details><summary> <b> External Tool Preparation </b> </summary>
<p>

We use 3DMM coefficients to represent a 3D listener or speaker, and for further 3D-to-2D frame rendering. 
 
The baselines leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients, and render 3D facial reactions.  

- You should first download 3DMM (FaceVerse version 2 model) at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and then put it in the folder (`external/FaceVerse/data/`).
 
  We provide our extracted 3DMM coefficients (which are used for our baseline visualisation) at [Google Drive] (https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu). 

  We also provide the mean_face, std_face and reference_full of 3DMM coefficients at [Google Drive](https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4). Please put them in the folder (`external/FaceVerse/`).

 
Then, we use a 3D-to-2D tool [PIRender](https://github.com/RenYurui/PIRender) to render final 2D facial reaction frames.
 
- We re-trained the PIRender, and the well-trained model is provided at the [checkpoint](https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA). Please put it in the folder (`external/PIRender/`).
   
</p>
</details>



<details><summary> <b> Training </b>  </summary>
<p>

- Running the following shell can start training ReactFace:
 ```shell
python train.py   --batch-size 4  --window-size 64  --momentum 0.1  --gpu-ids 0   -lr 0.00002   -e 200  -j 10  --kl-p 0.00001  --outdir results/training-reactface --div-p 100 --if-visual --if-audio  --if-vim --if-mim
 ```

</p>
</details>


<details><summary> <b> Evaluation </b>  </summary>
<p>

- Running the following shell can evaluate trained ReactFace:
 ```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch-size 16 --momentum 0.999  --window-size 8 --gpu-ids 0 --if-visual --if-audio  --if-vim --if-mim --outdir results/test-reactface  --resume results/training-reactface/best_checkpoint.pth
 ```

</p>
</details>


## 🖊️ Citation
=
if the code or method help you in the research, please cite the following paper:
```
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


## 🤝 Acknowledgement
Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) &#8194;

- [PIRender](https://github.com/RenYurui/PIRender) &#8194;



