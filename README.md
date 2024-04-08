# DFSS-IQA
Code for "Deep Feature Statistics Mapping for Generalized Screen Content Image Quality Assessment".  
![image](https://github.com/Baoliang93/DFSS-IQA/blob/main/DFSS_Release/framework.png) 

# Environment
* python=3.8.5
* pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0

# Running
* Data Prepare
- [x] Download the SCID and SIQAD datasets into the path: `./DFSS-IQA/datasets/`
- [x] We provide the pre-trained checkpoints [here](https://mega.nz/folder/3aBEjYCL#AoE23fKfc_Iw-PPQDXGpxA). You can download it and put the included  files into the path: `"./DFSS-IQA/DFSS_Release/models/"`. 

* Train: 
  - For Intra-dataset:
   -  SIQAD: `python iqaScrach.py --list-dir='../sci_scripts/siqad-scripts-6-2-2/' --pro=split_id  --resume='../models/siqad/checkpoint_latest.pkl' --dataset='IQA'`
   -  SCID:  `python python iqaScrach.py --list-dir='../sci_scripts/scid-scripts-6-2-2/' --pro=split_id  --resume='../models/scid/checkpoint_latest.pkl' --dataset='SCID' --n-dtype=46`
      - split_id: 0-9

  - For Cross-dataset:
   -  SIQAD: `python iqaScrach.py --list-dir='../sci_scripts/siqad-scripts-all/' --pro=0 --resume='../models/siqad-all/checkpoint_latest.pkl' --dataset='IQA'`
   -  SCID:  `python iqaScrach.py --list-dir='../sci_scripts/scid-scripts-all/' --pro=0 --resume='../models/scid-all/checkpoint_latest.pkl' --dataset='SCID' --n-dtype=46`
      
* Test:  
  - Intra-dataset: `python iqaIntraTest.py`
  - Cross-dataset: `python iqaCrossTest.py`
  - Demo: `python demo.py`

