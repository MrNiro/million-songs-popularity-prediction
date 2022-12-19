# million-songs-popularity-prediction

This is project aims at exploring Parallel Technics and Performance but not Machine Learning.

### Data Source
http://millionsongdataset.com/
http://millionsongdataset.com/pages/getting-dataset#subset

### Preprocessing
- You may need to modify the filepath in the source file
    - 别问我为什么不传参，因为我懒呀
- load data from h5 files, extract useful features and save as csv
    ```bash
    python src/data_processing.py
    ```
- Use BERT to enhance raw features
    - transfer Text Features to Embeddings
    - this may not be helpful to the prediction accuracy, just for dataset enhancement
    ```bash
    python src/npl_features.csv
    ```

### Training
- make sure "./processed/enhanced_data.csv" exist
- ```
  python src/fit_model.py
  ```

### Working on Discovery Cluster
```bash
# check available nodes
scontrol show res=csye7105
squeue -a -w NODE
srun --partition=reservation --reservation=csye7105 -w c0401 --pty --export=ALL  /bin/bash

module load cuda/11.7
conda info --env
source activate MY_ENV
# "conda activate MY_ENV" may also work

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda insatll xgboost scikit-learn pandas -y
```
