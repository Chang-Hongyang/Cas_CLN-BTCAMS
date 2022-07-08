python main_mhs.py --input raw_data/ --test_src raw_data/test_data.json --output output_models/out_put_16/ --bert_model  /home/tfguan/bert/chinese_roberta_wwm_ext_pytorch/ --train_mode test --res_path test.json
python cmeie_official_evaluation.py --golden_file raw_data/test_data.json --predict_file user_data/res_data/test_mhs.json
python main_mhs.py --input raw_data/ --test_src raw_data/val_data.json --output output_models/out_put_16/ --bert_model  /home/tfguan/bert/chinese_roberta_wwm_ext_pytorch/ --train_mode test --res_path val.json
python cmeie_official_evaluation.py --golden_file raw_data/val_data.json --predict_file user_data/res_data/val_mhs.json
