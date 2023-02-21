#!/bin/bash
#SBATCH --job-name=bioasq-pubmedbert-base-embeddings
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/RetrievalAugmentedClassification/bioasq-pubmedbert-base-embeddings.txt
#SBATCH --time=8:00:00

module load miniconda/4.12.0
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
MODEL_PATH='data/bioasq-l2/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
OUTPUT_DIR='bioasq-biomedbert-base-embeddings'
DATASET_NAME='bioasq-l2'
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

python retriever/apply_retriever.py \
  --dataset_name ${DATASET_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --model_name ${MODEL_PATH}


