# DFSS-IQA
Code for "Deep Feature Statistics Mapping for Generalized Screen Content Image Quality Assessment".  
![image](https://github.com/Baoliang93/DFSS-IQA/blob/main/DFSS_Release/framework.png) 

# Environment
* python=3.8.5
* pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0

# Running
* Data Prepare
- [x] Download the SCID and SIQAD datasets into the path: `./DFSS-IQA/datasets/`
- [x] We provide the pretrained checkpoints [here](https://mega.nz/folder/iDxH3R6a#WF25kk1XD30fhlZeSPJzDA). You can download it and put the included  files into the path: `"./DFSS-IQA/DFSS_Release/models"`. 

* Train: 
  - For NI:  
    `python ./FPR/FPR_IQA/FPR_SCI/src/iqaScrach.py --list-dir='../scripts/dataset_name/' --resume='../models/model_files/checkpoint_latest.pkl' --pro=split_id --dataset='dataloader_name'`  
      -    dataset_name: "tid2013", "databaserelease2", "CSIQ", or "kadid10k"  
      -    model_files: "tid2013", "live", "csiq", or "kadid"
      - dataloader_name: "IQA" (for live and csiq  datasets), "TID2013", or "KADID"  
      - split_id: '0' to '9'
  - For SCI:   
      -  SIQAD: `python ./FPR/FPR_IQA/FPR_SCI/src/iqaScrach.py  --pro=split_id`    
      -  SCID: `python ./FPR/FPR_IQA/FPR_SCI/src/scid-iqaScrach.py  --pro=split_id`   
      
* Test:  
  - For NI:   
  `python ./FPR/FPR_IQA/FPR_SCI/src/iqaTest.py --list-dir='../scripts/dataset_name/' --resume='../models/model_files/model_best.pkl' --pro=split_id  --dataset='dataloader_name'`  
   - For SCI:   
      -  SIQAD: `python ./FPR/FPR_IQA/FPR_SCI/src/iqaTest.py  --pro=split_id`    
      -  SCID: `python ./FPR/FPR_IQA/FPR_SCI/src/scid-iqaTest.py  --pro=split_id`
