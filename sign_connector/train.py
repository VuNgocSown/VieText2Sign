"""Training script for Sign Connector model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import numpy as np
import queue
import logging
from tqdm import tqdm
from datetime import datetime
import yaml

from .model import SignConnector
from .dataset import ConnectorDataset, remap_skeletons, SKELETONS
from .utils import set_seed


def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = os.path.join(log_dir, f"connector_train_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logf), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, criterion, device, edge_index, edge_weight):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        x = batch['kps_input'].to(device)
        y = batch['labels'].to(device)
        
        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)
        loss = criterion(out, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, edge_index, edge_weight):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['kps_input'].to(device)
            y = batch['labels'].to(device)
            out = model(x, edge_index, edge_weight)
            total_loss += criterion(out, y.unsqueeze(1)).item()
    
    return total_loss / len(val_loader)


def train_connector(config_path=None):
    """Main training function."""
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config['paths']['logs'])
    logger.info("🚀 Training Sign Connector Model...")
    logger.info(f"Config: {config_path}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Device setup
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Joint indices
    joint_idx = np.array(config['data']['joint_indices'], dtype=np.int32)
    
    # Create datasets
    root_dir = config['data']['root_dir']
    kps_path = os.path.join(root_dir, config['data']['keypoints_file'])
    train_pairs = os.path.join(root_dir, config['data']['train_pairs'])
    val_pairs = os.path.join(root_dir, config['data']['val_pairs'])
    
    train_dataset = ConnectorDataset(kps_path, train_pairs, joint_idx)
    val_dataset = ConnectorDataset(kps_path, val_pairs, joint_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=ConnectorDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=ConnectorDataset.collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = SignConnector(
        num_joints=config['model']['num_joints'],
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    )
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup skeleton edges
    skeletons_local = remap_skeletons(SKELETONS, joint_idx)
    edge_index = torch.tensor(skeletons_local, dtype=torch.long).t().contiguous().to(device)
    edge_weight = torch.ones(edge_index.shape[1], device=device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=config['training']['scheduler']['eta_min'],
        T_max=config['training']['scheduler']['T_max']
    )
    
    criterion = nn.L1Loss()
    
    # Training loop
    best_val = float('inf')
    best_ep = 0
    checkpoint_dir = config['paths']['checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_q = queue.Queue(maxsize=1)
    
    for epoch in tqdm(range(config['training']['epochs'])):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, edge_index, edge_weight
        )
        scheduler.step()
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion,
            device, edge_index, edge_weight
        )
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_ep = epoch
            fname = os.path.join(checkpoint_dir, f'connector_ep{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, fname)
            logger.info(f"✅ Saved best model: {fname}")
            
            if ckpt_q.full():
                old_file = ckpt_q.get()
                if os.path.exists(old_file):
                    os.remove(old_file)
            ckpt_q.put(fname)
        
        logger.info(
            f"Epoch {epoch:03d} | Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | Best: {best_val:.6f} (Ep {best_ep})"
        )
    
    logger.info(f"✅ Training completed! Best validation loss: {best_val:.6f} at epoch {best_ep}")
    return best_val, best_ep


if __name__ == "__main__":
    train_connector()
