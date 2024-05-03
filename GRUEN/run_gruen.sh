# Preprocessing script for your dataset
mkdir -p ./output/

python3 ./dataset_split_gruen.py \
--input_dir='../DFTData/Latest/*' \
--output_dir='./input/' \
--grover_ds=0 \
--len_disc=0  


# Change to path where GRUEN is installed

# Correct format
python3 Main_l.py \
--input_dir='./input/*' \
--cola_dir='./cola_model/bert-base-cased/' \
--cache_dir='./models' \
--label='machine' \
--discriminator_type='gltr' \
--output_dir='./output/'
   