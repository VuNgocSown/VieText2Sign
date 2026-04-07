import torch
import yaml
from pathlib import Path


ACTIVE_EXPERIMENT = "nllb"

CONFIG_DIR = Path(__file__).parent / "configs"


def load_experiment_config(experiment_name: str) -> dict:
    """Load experiment configuration from YAML file"""
    config_file = CONFIG_DIR / f"{experiment_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_active_config() -> dict:
    """Get configuration for active experiment"""
    return load_experiment_config(ACTIVE_EXPERIMENT)


def list_available_experiments() -> list:
    """List all available experiment configurations"""
    return [f.stem for f in CONFIG_DIR.glob("*.yaml")]


class Config:
    def __init__(self, experiment_name: str = None):
        if experiment_name is None:
            experiment_name = ACTIVE_EXPERIMENT
        
        self.experiment_name = experiment_name
        self.config = load_experiment_config(experiment_name)
        
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Config(experiment={self.experiment_name}, model={self.model_checkpoint})"


DATA_DIR = Path(__file__).parent / "data"
MODEL_SAVE_DIR = Path(__file__).parent.parent / "models" / "text2gloss"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
SEED = 42
