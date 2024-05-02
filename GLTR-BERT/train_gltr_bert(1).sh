export CUDA_VISIBLE_DEVICES=1

python3 -u gltr_train_bert.py \
--train_dataset='../DFTData/Perturbed_Train_Data_300.jsonl' \
--bert_large_gltr_ckpt='./ckpts/gltrbert_dftTrain300Model.sav' \
--output_metrics='./metrics/gltrbert_perturbed_Train300.jsonl' \
--bert_model='bert-large-cased' \
>./logs/gltrbert_perturbed_Train300.txt 



