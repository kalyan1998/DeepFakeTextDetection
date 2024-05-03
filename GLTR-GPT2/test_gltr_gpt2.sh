export CUDA_VISIBLE_DEVICES=0,1

python3 -u gltr_test_gpt2xl.py \
--test_dataset='../Data/WildData/GLTR_GPT2_BERT_test.jsonl' \
--gpt2_xl_gltr_ckpt='./ckpts/gltrgpt2_dftTrain300Model.sav' \
--gpt2_model='gpt2-xl' \
--return_stat_file='./hist_stats/wildDataTestonDFTmodel_GLTRGPT2.jsonl' \
--output_metrics='./metrics/wildDataTestonDFTmodel_GLTRGPT2.jsonl' \
>./logs/wildDataTestonDFTmodel_GLTRGPT2.txt
