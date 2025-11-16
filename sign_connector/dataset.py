"""Dataset and data processing utilities for Sign Connector."""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


# Skeleton connections
SKELETONS = [
    (2,3),(3,46),(46,47),(47,48),(48,49),(49,50),
    (46,51),(51,52),(52,53),(53,54),(46,55),(55,56),(56,57),(57,58),
    (46,59),(59,60),(60,61),(61,62),(46,63),(63,64),(64,65),(65,66),
    (5,6),(6,25),(25,26),(26,27),(27,28),(28,29),(25,30),(30,31),(31,32),
    (32,33),(25,34),(34,35),(35,36),(36,37),(25,38),(38,39),(39,40),(40,41),
    (25,42),(42,43),(43,44),(44,45)
]


def remap_skeletons(skeletons, joint_idx):
    """Remap skeleton connections to match selected joint indices."""
    idx_map = {j: i for i, j in enumerate(joint_idx)}
    mapped = []
    for p, c in skeletons:
        if p in idx_map and c in idx_map:
            mapped.append((idx_map[p], idx_map[c]))
    return mapped


def make_7d_for_joints(kps, skeletons):
    """Convert 3D keypoints to 7D (position + bone length + direction)."""
    B, N, _ = kps.shape
    out = torch.zeros(B, N, 7, device=kps.device)
    out[:, :, :3] = kps
    if len(skeletons) == 0:
        return out

    parents = torch.tensor([p for p, c in skeletons], device=kps.device)
    childs = torch.tensor([c for p, c in skeletons], device=kps.device)
    parent_pos = kps[:, parents, :]
    child_pos = kps[:, childs, :]
    v = child_pos - parent_pos
    length = torch.norm(v, dim=2, keepdim=True)
    direction = v / (length + 1e-8)
    bone_feat = torch.cat([length, direction], dim=2)
    out[:, childs, 3:] = bone_feat
    return out


class ConnectorDataset(Dataset):
    """Dataset for training sign connector with temporal differences."""
    def __init__(self, kps_path, pairs_path, joint_idx):
        with open(kps_path, 'rb') as f:
            self.kps = pickle.load(f)
        with open(pairs_path, 'rb') as f:
            self.pairs = pickle.load(f)
        self.joint_idx = joint_idx
        self.skeletons_local = remap_skeletons(SKELETONS, joint_idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clip1, clip2 = self.pairs[idx]
        end, start = clip1['end'], clip2['start']
        label = torch.tensor(start - end).float()

        kps1 = torch.tensor(
            self.kps[clip1['video_file']]['keypoints_3d'][end][self.joint_idx]
        ).float().unsqueeze(0)
        kps2 = torch.tensor(
            self.kps[clip2['video_file']]['keypoints_3d'][start][self.joint_idx]
        ).float().unsqueeze(0)
        
        kps1_7d = make_7d_for_joints(kps1, self.skeletons_local).squeeze(0)
        kps2_7d = make_7d_for_joints(kps2, self.skeletons_local).squeeze(0)

        delta = kps2_7d - kps1_7d
        kps_input = torch.cat([kps1_7d, delta], dim=1)
        return {'kps_input': kps_input, 'labels': label}

    @staticmethod
    def collate_fn(batch):
        kps_input = [s['kps_input'] for s in batch]
        label = [s['labels'] for s in batch]
        return {
            'kps_input': torch.stack(kps_input, dim=0),
            'labels': torch.stack(label, dim=0)
        }
