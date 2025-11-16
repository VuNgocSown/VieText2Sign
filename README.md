# Text2Sign Pipeline: Vietnamese Text to Sign Language Translation with 3D Avatars

A complete pipeline for translating Vietnamese text into sign language videos using 3D avatars.

## Introduction

This pipeline enables Vietnamese text to sign language (Text2Sign) translation. The system consists of three main components:
1. **Text2Gloss Translator**: Converting Vietnamese text to sign language gloss notation using fine-tuned mBART-large-50
2. **Sign Connector**: Predicting optimal transition frames between sign poses using GCN-based model
3. **Rendering Module**: Generating sign language videos with SMPL-X 3D avatars

The translation results are displayed through realistic sign avatars rendered using Blender.

## Environment

Please run:
```bash
pip install -r requirements.txt
```

Or you may use conda environment:
```bash
conda create -n text2sign python=3.10
conda activate text2sign
pip install -r requirements.txt
```

## Data Preparation

### Vietnamese Sign Gloss Corpus
The training data should be an Excel file with two columns:
- Column 1: Vietnamese text
- Column 2: Corresponding gloss notation

Place the data file at `text2gloss/data/Corpus-Vie-VSL-10K.xlsx`.

### Keypoints
Keypoints data for sign connector training should be in pickle format containing 3D joint positions.
Place files at:
- `sign_connector/data/keypoints_3d_mesh.pkl`
- `sign_connector/data/iso_clip_pairs.train`
- `sign_connector/data/iso_clip_pairs.dev`

### SMPL-X Model
Please download SMPL-X models and place them at `models/data/smplx/`.
The structure should include:
```
models/data/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
└── SMPLX_FEMALE.npz
```

### SMPL-X Blender Add-on
Rendering 3D avatars relies on [Blender](https://www.blender.org/download/) and SMPL-X add-on.
Place the add-ons at `pretrained_models/smplx_blender_addon/`.

### Pre-trained Models
Download pre-trained models:
- Text2Gloss model: Place at `models/text2gloss/`
- Sign Connector: Place at `models/data/connector.pth`

## Training

### Text2Gloss Translator
To train the text2gloss translator, run:
```bash
python -m text2gloss.train
```


### Sign Connector
To train the sign connector, run:
```bash
python -m sign_connector.train
```

## Text2Sign Translation

### Step 1: Text to Gloss Translation
Run text2gloss prediction:
```bash
python -m text2gloss.test_result
```

Or use programmatically:
```python
from text2gloss_predictor import Text2GlossPredictor

predictor = Text2GlossPredictor('models/text2gloss')
gloss = predictor.predict("xin chào các bạn")
print(f"Gloss: {gloss}")
```

### Step 2: Complete Pipeline
Generate sign language video by running:
```bash
python main.py
```



## Evaluation

### Text2Gloss Evaluation
To evaluate the text2gloss model:
```bash
python -m text2gloss.evaluation
```



## Project Structure

```
text2sign_pipeline/
├── sign_connector/              # Sign connector module
│   ├── train.py                # Training script
│   ├── model.py                # SignConnector model
│   ├── dataset.py              # Dataset loader
│   ├── utils.py                # Utility functions
│   ├── configs/
│   │   └── config.yaml         # Training configuration
│   ├── data/                   # Training data
│   ├── checkpoints/            # Model checkpoints
│   └── logs/                   # Training logs
│
├── text2gloss/                 # Text2gloss module
│   ├── train.py                # Training script
│   ├── evaluation.py           # Evaluation script
│   ├── test_result.py          # Interactive prediction
│   ├── config.py               # Configuration
│   ├── preprocess.py           # Data preprocessing
│   ├── metrics.py              # Evaluation metrics
│   ├── model_utils.py          # Model utilities
│   ├── logger_utils.py         # Logging utilities
│   ├── data/                   # Training corpus
│   ├── models/                 # Saved models
│   └── logs/                   # Training logs
│
├── models/                     # Pre-trained models
│   ├── data/
│   │   ├── connector.pth       # Sign connector weights
│   │   └── smplx/              # SMPL-X body models
│   └── text2gloss/             # Text2gloss model
│
├── pretrained_models/          # External dependencies
│   └── smplx_blender_addon/    # Blender add-on
│
├── vposer/                     # VPoser model
│
├── scripts/                    # Utility scripts
│   └── render_avatar.py        # Avatar rendering
│
├── pipeline.py                 # Main pipeline orchestration
├── main.py                     # Entry point
├── text2gloss_predictor.py     # Text2gloss inference wrapper
├── sign_connector_wrapper.py   # Sign connector inference wrapper
├── blender_renderer.py         # Blender rendering utilities
├── video_creator.py            # Video generation
├── utils.py                    # Common utilities
├── config.json                 # Pipeline configuration
└── requirements.txt            # Python dependencies
```

