#!/bin/bash
#SBATCH --job-name=bioasq-ra-labels-pubmedbert-base
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/RetrievalAugmentedClassification/bioasq-ra-labels-pubmedbert-base.txt
#SBATCH --time=8:00:00

module load miniconda/4.12.0
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
MODEL_PATH='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
EMBEDDINGS_FOLDER='bioasq-biomedbert-base-embeddings'
DATASET_NAME='bioasq-l2'
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
rm -rf ../.cache/huggingface/datasets/kiddothe2b___multilabel_bench/${DATASET_NAME}
python classifier/train_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --embeddings_path ${EMBEDDINGS_FOLDER} \
    --retrieval_augmentation true \
    --augment_with_labels true \
    --augment_with_documents false \
    --encode_document true \
    --no_neighbors 32 \
    --dataset_name ${DATASET_NAME} \
    --output_dir data/${DATASET_NAME}/${MODEL_PATH}-ra-labels \
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
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --seed 42 \
    --num_train_epochs 20 \
    --max_train_samples 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2
