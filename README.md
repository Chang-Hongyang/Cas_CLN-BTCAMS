# Cas_CLN-BTCAMS
Cas-CLN ==> *mpn*   
BTCAMS ==> *mhs*  

pretrained model：  
|----chinese_roberta_wwm_large_ext_pytorch  

raw_data:  
	|----test_data.json  
	|----train_data.json  
	|----val_data.json  
user_data:  
  |----split_data:  
    |----|----data_set1:  
      |----|----|----train_data.json  
      |----|----|----val_data.json  

requirement:  
  ----torch==1.2.0  
  ----tqdm==4.35.0  
  ----transformers==2.2.2  
  
mkdir output_models  
train: train_*.sh  
test: test_*.sh  
