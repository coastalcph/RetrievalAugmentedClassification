Requirements: `transformers`, `pytorch`, `sentence-transformers`

Run with: 
```
python train_retriever.py --data_path <FILEPATH> --experiment_name <EXP_NAME>
```

where FILEPATH is a path to a text file (optionally compressed with gzip), where each line contain either a single document (for unsupervised training) or two tab-separated documents (for supervised training).
