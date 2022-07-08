# Cas_CLN-BTCAMS
cd Cas_CLN&BTCAMS  
Cas-CLN ==> *mpn*   
BTCAMS ==> *mhs*  

pretrained model：  
|----chinese_roberta_wwm_large_ext_pytorch   
(You should download by yourself from google. The folder is too large. https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)  

data:   
    CMeIE is also public available. Guan, T., Zan, H., Zhou, X., Xu, H., Zhang, K. (2020). CMeIE: Construction and Evaluation of Chinese Medical Information Extraction Dataset. In: Zhu, X., Zhang, M., Hong, Y., He, R. (eds) Natural Language Processing and Chinese Computing. NLPCC 2020. Lecture Notes in Computer Science(), vol 12430. Springer, Cham. https://doi.org/10.1007/978-3-030-60450-9_22  
    You can use your data by following the CMeIE's format or adjusting the code/spo_*/data_loader.py.  
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
train: bash train_*.sh  
test: bash test_*.sh  
