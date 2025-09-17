# Rugby CV Project Infrastructure Setup
This is the original setup suggestion from Claude, it is not fully implemented yet, and changes are being made to the design as I work through problems. I'm storing it here in it's entirety for future reference.

## Overview
This guide sets up a production-ready ML infrastructure for the rugby tackle detection project using AWS, Docker, Python scripts, and experiment tracking.

## Architecture

```
rugby-cv-project/
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.inference
│   └── docker-compose.yml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── supervisely_parser.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── movinet.py
│   │   └── base_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── config.py
│   │   └── utils.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── video_processor.py
│   └── experiments/
│       ├── __init__.py
│       └── experiment_tracker.py
├── configs/
│   ├── base_config.yaml
│   ├── movinet_config.yaml
│   └── aws_config.yaml
├── scripts/
│   ├── setup_aws.sh
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess_data.py
│   └── run_experiment.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── experiments/
├── .dvc/
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## 1. AWS Setup

### EC2 Instance for Training
- **Instance Type**: `g4dn.xlarge` (1 GPU, good for MoViNet training)
- **AMI**: Deep Learning AMI (Ubuntu 20.04)
- **Storage**: 100GB EBS GP3

### S3 Buckets
- `rugby-cv-data`: Raw videos, annotations, processed datasets
- `rugby-cv-models`: Trained models, experiment artifacts
- `rugby-cv-logs`: Training logs, experiment tracking

### AWS CLI Setup Script
```bash
#!/bin/bash
# scripts/setup_aws.sh

# Configure AWS CLI
aws configure

# Create S3 buckets
aws s3 mb s3://rugby-cv-data
aws s3 mb s3://rugby-cv-models
aws s3 mb s3://rugby-cv-logs

# Set up IAM role for EC2 access to S3
aws iam create-role --role-name rugby-cv-role --assume-role-policy-document file://aws-trust-policy.json
aws iam attach-role-policy --role-name rugby-cv-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## 2. Docker Setup

### Training Dockerfile
```dockerfile
# docker/Dockerfile.train
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install project in development mode
COPY setup.py .
COPY src/ ./src/
RUN pip install -e .

# Copy scripts
COPY scripts/ ./scripts/
COPY configs/ ./configs/

EXPOSE 8888
CMD ["bash"]
```

### Docker Compose
```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  training:
    build:
      context: ..
      dockerfile: docker/Dockerfile.train
    volumes:
      - ../data:/app/data
      - ../experiments:/app/experiments
      - ../models:/app/models
      - ~/.aws:/root/.aws:ro
    environment:
      - AWS_PROFILE=default
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "8888:8888"
    command: ["python", "scripts/train.py", "--config", "configs/movinet_config.yaml"]
    
  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile.train
    volumes:
      - ../:/app
      - ~/.aws:/root/.aws:ro
    ports:
      - "8888:8888"
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

## 3. DVC Setup

### Initialize DVC
```bash
cd rugby-cv-project
git init
dvc init
dvc remote add -d storage s3://rugby-cv-data/dvc-cache
```

### DVC Pipeline (dvc.yaml)
```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess_data.py
    deps:
    - data/raw/
    - src/data/preprocessing.py
    params:
    - preprocess.clip_length
    - preprocess.frame_size
    outs:
    - data/processed/clips/
    
  train:
    cmd: python scripts/train.py --config configs/movinet_config.yaml
    deps:
    - data/processed/clips/
    - src/models/movinet.py
    - src/training/trainer.py
    params:
    - training.batch_size
    - training.epochs
    - training.learning_rate
    outs:
    - models/movinet_latest.h5
    metrics:
    - experiments/metrics.json
    
  evaluate:
    cmd: python scripts/evaluate.py --model models/movinet_latest.h5
    deps:
    - models/movinet_latest.h5
    - data/processed/test/
    metrics:
    - experiments/evaluation.json
```

### Parameters (params.yaml)
```yaml
preprocess:
  clip_length: 40
  frame_size: [224, 224]
  test_split: 0.2
  
training:
  model_name: "movinet_a0"
  batch_size: 8
  epochs: 50
  learning_rate: 0.001
  freeze_backbone: true
  
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
```

## 4. Core Python Scripts

### Training Script
```python
# scripts/train.py
import argparse
import yaml
from pathlib import Path
import mlflow
import tensorflow as tf
from src.training.trainer import Trainer
from src.data.dataset import RugbyDataset
from src.models.movinet import MoviNetModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--experiment-name', default='rugby-tackle-detection')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config['training'])
        
        # Create dataset
        dataset = RugbyDataset(config)
        train_ds, val_ds = dataset.get_datasets()
        
        # Create model
        model = MoviNetModel(config['training'])
        
        # Train
        trainer = Trainer(model, config)
        history = trainer.train(train_ds, val_ds)
        
        # Log metrics
        for epoch, metrics in enumerate(history.history):
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value, step=epoch)
        
        # Save model
        model_path = f"models/{mlflow.active_run().info.run_id}.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
```

### Experiment Runner
```python
# scripts/run_experiment.py
import itertools
import yaml
from pathlib import Path
import subprocess
import json

def run_experiment_grid():
    """Run hyperparameter grid search."""
    
    # Define parameter grid
    param_grid = {
        'training.learning_rate': [1e-3, 1e-4, 1e-5],
        'training.freeze_backbone': [True, False],
        'training.batch_size': [8, 16],
        'preprocess.clip_length': [40, 60, 80]
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combination in itertools.product(*values):
        # Create config for this experiment
        config = create_config_from_params(dict(zip(keys, combination)))
        
        # Save config
        config_path = f"configs/experiment_{hash(str(combination))}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run experiment
        print(f"Running experiment with config: {config_path}")
        subprocess.run([
            "python", "scripts/train.py", 
            "--config", config_path,
            "--experiment-name", "hyperparameter-search"
        ])

if __name__ == "__main__":
    run_experiment_grid()
```

## 5. Experiment Tracking with MLflow

### Setup MLflow
```bash
# Run MLflow server
mlflow server --backend-store-uri sqlite:///experiments/mlflow.db --default-artifact-root s3://rugby-cv-models/mlflow-artifacts --host 0.0.0.0
```

### Track Experiments
```python
# src/experiments/experiment_tracker.py
import mlflow
import matplotlib.pyplot as plt
import json

class ExperimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
    
    def log_model_performance(self, model, test_dataset, run_name=None):
        with mlflow.start_run(run_name=run_name):
            # Evaluate model
            predictions = model.predict(test_dataset)
            metrics = self.calculate_metrics(predictions, test_dataset)
            
            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            # Log confusion matrix
            self.log_confusion_matrix(predictions, test_dataset)
    
    def compare_experiments(self, experiment_ids):
        """Compare multiple experiment runs."""
        for exp_id in experiment_ids:
            run = mlflow.get_run(exp_id)
            print(f"Run {exp_id}: Accuracy = {run.data.metrics['accuracy']:.3f}")
```

## 6. Getting Started

### Initial Setup
```bash
# Clone and setup
git clone <your-repo>
cd rugby-cv-project

# Build Docker containers
docker-compose -f docker/docker-compose.yml build

# Initialize DVC
dvc pull  # Get data from S3

# Run first experiment
docker-compose -f docker/docker-compose.yml run training python scripts/train.py --config configs/movinet_config.yaml
```

### Development Workflow
```bash
# 1. Update parameters
vim params.yaml

# 2. Run DVC pipeline
dvc repro

# 3. Compare experiments
mlflow ui

# 4. Deploy best model
dvc metrics diff
```

## Next Steps

1. **Set up AWS infrastructure**
2. **Create Docker containers**
3. **Migrate your existing data to S3**
4. **Run first training experiment**
5. **Set up MLflow for experiment tracking**
6. **Create hyperparameter search grid**

This infrastructure will give you:
- ✅ Reproducible experiments
- ✅ Version-controlled data and models
- ✅ Cloud-based training
- ✅ Proper experiment tracking
- ✅ Easy scaling and collaboration