# Activate virtual environment
source roberta/bin/activate

# RoBERTa-Defense fine-tuning script
# Important parameter description can be found by ``python xx.py -h''
TRAIN_DATA="../DFTData/train_data_perturbed.jsonl"
VAL_DATA="../DFTData/val_data_perturbed.jsonl"
TEST_DATA="../DFTData/test_data_perturbed.jsonl"
OUTPUT_DIR="./model_ckpts"
BASENAME="robertaTrainDFTData_300"

export CUDA_VISIBLE_DEVICES=0,1  # Adjust as per your GPU setup
python3 -u ./roberta_finetune.py \
--cache_dir='./models' \
--train_dir="${TRAIN_DATA}" \
--val_dir="${VAL_DATA}" \
--test_dir="${TEST_DATA}" \
--prediction_output="./metrics/${BASENAME}_metrics.jsonl" \
--output_dir="${OUTPUT_DIR}" \
--logging_file="./logs/${BASENAME}_logging.txt" \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=1 \
--val_batch_size=1 \
--token_len=512 \
--model_ckpt_path='./Checkpoint/checkpoint-full' \
--num_train_epochs=6 \
--save_steps=30 \
> ./logs/${BASENAME}.txt 
