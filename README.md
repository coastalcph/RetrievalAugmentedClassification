# RetrievalAugmentedClassification


Requirements: `transformers`, `pytorch`, `sentence-transformers`, `datasets`

# Train Retriever

Run with: 
```
python train_retriever.py --dataset_name <e.g. ecthr> --experiment_name <EXP_NAME>
```


# Fine-tuning

To fine-tune a model, run:

```shell
#!/bin/bash
#SBATCH --job-name=ecthr-longformer
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/RetrievalAugmentedClassification/ecthr-longformer.txt
#SBATCH --time=8:00:00

module load miniconda/4.12.0
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES
MODEL_PATH='allenai/longformer-base-4096'
DATASET_NAME='ecthr'

python train_classifier \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ../data/${DATASET_NAME}/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --max_seq_length 2048 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --seed 42 \
    --num_train_epochs 20 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --lr_scheduler_type cosine
```

# Create custom Longformer

To create and use a custom Longformer-based run `convert_roberta_to_lf` for RoBERTa models and `convert_bert_to_lf` for BERT models:

```shell
module load miniconda/4.12.0
conda activate kiddothe2b

ROBERTA_MODEL_PATH='lexlms/roberta-large-cased'
OUTPUT_MODEL_PATH='lexlms/longformer-large'

python utils/convert_roberta_to_lf.py \
--roberta_checkpoint ${ROBERTA_MODEL_PATH} \
--output_model_path ${OUTPUT_MODEL_PATH}
```

then use `MODEL_PATH='data/lexlms/longformer-large'` to fine-tune this model.

For legal tasks:
- `lexlms/roberta-large-cased`
- `lexlms/roberta-base-cased`

For biomedical tasks:
- `bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16`
- `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12`
