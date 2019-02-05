# export CUDA_VISIBLE_DEVICES=1
python3 src/main.py \
--epoch 100 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 35 \
--save_itr 250 \
--sample_size 30 \
--input_height 256 \
--output_height 256 \
--lambda_val 1.0 \
--smoothing 1.0 \
--use_resize True \
--dataset wikiart \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop False \
--use_s3 \
--s3_bucket "creative-adv-nets" \
--can True \
--train \
