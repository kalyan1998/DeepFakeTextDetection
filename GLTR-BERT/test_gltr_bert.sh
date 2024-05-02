export CUDA_VISIBLE_DEVICES=1

python3 -u gltr_test_bert.py \
--test_dataset='../Data/WildData/GLTR_GPT2_BERT_test.jsonl' \
--bert_model='bert-large-cased' \
--bert_large_gltr_ckpt='./ckpts/gltrbert_dftTrain300Model.sav' \
--return_stat_file="./hist_stats/wildDataTestonDFTmodel_GLTRBERT.jsonl" \
--output_metrics="./metrics/wildDataTestonDFTmodel_GLTRBERT.jsonl" \
> ./logs/wildDataTestonDFTmodel_GLTRBERT.txt

