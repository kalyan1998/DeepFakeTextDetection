# BERT-Defense evaluation script
# Important parameter description can be found by ``python xx.py -h''
export CUDA_VISIBLE_DEVICES=0,1
python3 -u bert_defense_eval.py \
--cache_dir='./models' \
--test_dir='../BERT-Defense_test.jsonl' \
--prediction_output='./wildDataDFTModelTest_BERT.jsonl' \
--output_dir='./ckpts' \
--logging_file='./logs/wildDataDFTModelTest_BERT.jsonl ' \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=1 \
--val_batch_size=1 \
--model_ckpt_path='./ckpts/checkpoint-840' \
--num_train_epochs=6 \
--save_steps=1000 \
>./logs/wildDataDFTModelTest_BERT.txt