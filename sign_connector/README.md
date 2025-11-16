# Sign Connector

Training module for predicting transition frames between sign language poses.

## Structure

```
sign_connector/
├── __init__.py          # Module initialization
├── train.py            # Training script
├── model.py            # SignConnector model
├── dataset.py          # Dataset and preprocessing
├── utils.py            # Utility functions
├── configs/
│   └── config.yaml     # Training configuration
├── data/               # Training data
│   ├── keypoints_3d_mesh.pkl
│   ├── iso_clip_pairs.train
│   └── iso_clip_pairs.dev
├── checkpoints/        # Model checkpoints
└── logs/              # Training logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd /home/vnson/vnson/text2sign_pipeline
python -m sign_connector.train
```

Or from Python:

```python
from sign_connector import train_connector
train_connector()
```

## Configuration

Edit `configs/config.yaml`:
- **Model**: num_joints, hidden_dim, dropout
- **Training**: batch_size, learning_rate, epochs
- **Data**: paths to keypoints and pairs files

## Model

**SignConnector**: GCN-based architecture
- Input: 14D features per joint (7D pose + 7D temporal delta)
- 2 GCN layers with coordinate normalization
- Output: Predicted transition frames (scalar)

## Output

- Checkpoints: `checkpoints/connector_ep{epoch}.pth`
- Logs: `logs/connector_train_{timestamp}.log`
