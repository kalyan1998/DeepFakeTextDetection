TRAIN_DATA="../DFTData/train_data_perturbed.jsonl"
VAL_DATA="../DFTData/val_data_perturbed.jsonl"
TEST_DATA="../DFTData/test_data_perturbed.jsonl"
OUTPUT_DIR="./ckpts"
SAVE_NAME="bertTrainDFTData_300"


# BERT-Defense finetuning script
# Important parameter description can be found by ``python xx.py -h''
export CUDA_VISIBLE_DEVICES=0,1
python3 bert_fine_tune.py \
--cache_dir='./models' \
--train_dir=${TRAIN_DATA} \
--val_dir=${VAL_DATA} \
--test_dir=${TEST_DATA} \
--prediction_output="./metrics/${SAVE_NAME}.jsonl" \
--output_dir=${OUTPUT_DIR} \
--logging_file="./logs/${SAVE_NAME}.txt" \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=1 \
--val_batch_size=1 \
--token_len=512 \
--model_ckpt_path='./Checkpoint/checkpoint-14000' \
--num_train_epochs=8 \
--save_steps=30
