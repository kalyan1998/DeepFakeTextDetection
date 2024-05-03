export CUDA_VISIBLE_DEVICES=0,1

python3 -u gltr_train_gpt2xl.py \
--train_dataset='../DFTData/Perturbed_GPT_Train_Data_300.jsonl' \
--gpt2_xl_gltr_ckpt='./ckpts/gltrgpt2_dftTrain300Model.sav' \
--output_metrics='./metrics/gltrgpt2_perturbed_Train300.jsonl' \
--gpt2_model='gpt2-xl' \
>./logs/gltrgpt2_perturbed_Train300.txt 



