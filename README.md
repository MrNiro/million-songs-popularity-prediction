# million-songs-popularity-prediction

## Training
- make sure "./processed/enhanced_data.csv" exist
- go to ./src
    - ```
      python fit_model.py
      ```

## Working on Discovery Cluster
```bash
# check available nodes
scontrol show res=csye7105
squeue -a -w NODE
srun --partition=reservation --reservation=csye7105 -w c0401 --pty --export=ALL  /bin/bash

module load cuda/11.7
conda info --env
conda activate MY_ENV

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda insatll xgboost scikit-learn pandas -y
```
