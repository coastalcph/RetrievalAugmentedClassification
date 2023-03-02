#!/bin/bash
#SBATCH --job-name=bioasq-pubmedbert-base-bootstrap
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/RetrievalAugmentedClassification/bioasq-base-bootstrap.txt
#SBATCH --time=8:00:00

module load miniconda/4.12.0
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
MODEL_PATH='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
PREDICTIONS_FOLDER='bioasq-base-predictions'
DATASET_NAME='bioasq-l2'
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# DELETE DATASET
rm -rf ../.cache/huggingface/datasets/kiddothe2b___multilabel_bench/${DATASET_NAME}

# SAVE PREDICTIONS
python classifier/train_classifier.py \
    --model_name_or_path data/${DATASET_NAME}/${MODEL_PATH} \
    --retrieval_augmentation false \
    --bootstrap_dataset false \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${DATASET_NAME}/${PREDICTIONS_FOLDER} \
    --do_train false \
    --do_eval false \
    --do_pred false \
    --do_train_eval true \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --lr_scheduler_type cosine

# DELETE DATASET
rm -rf ../.cache/huggingface/datasets/kiddothe2b___multilabel_bench/${DATASET_NAME}

# TRAIN BOOTSTRAPPED CLASSIFIER
python classifier/train_classifier.py \
    --model_name_or_path data/${DATASET_NAME}/${MODEL_PATH} \
    --retrieval_augmentation false \
    --bootstrap_dataset true \
    --predictions_path ${DATASET_NAME}/${PREDICTIONS_FOLDER} \
    --dataset_name ${DATASET_NAME} \
    --output_dir data/${DATASET_NAME}/${MODEL_PATH}-bootstrap \
    --do_train \
    --do_eval \
    --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --lr_scheduler_type cosine