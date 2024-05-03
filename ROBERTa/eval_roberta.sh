# install virtual env and then activate virtual env
source roberta/bin/activate

# RoBERTa-Defense evaluation script
# Important parameter description can be found by ``python xx.py -h''
TEST_DATA="../Data/WildData/RoBERTa_Defense_test.jsonl"
BASENAME="wildDataTestonDFTmodel_Roberta"

export CUDA_VISIBLE_DEVICES=1
python3 -u ./roberta_defense_eval.py \
--cache_dir='./models' \
--test_dir="${TEST_DATA}" \
--prediction_output="./metrics/${BASENAME}.jsonl" \
--output_dir='./model/' \
--logging_file="./logging/${BASENAME}_logging.txt" \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=1 \
--val_batch_size=1 \
--model_ckpt_path='./model_ckpts/checkpoint-80' \
--num_train_epochs=6 \
--save_steps=5000 \
>./logs/${BASENAME}.txt 