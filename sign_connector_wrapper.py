"""Sign Connector Wrapper Module"""
import os
import pickle
import numpy as np
import torch
from numpy import pi
import smplx
from utils import smpl_to_openpose, JointMapper


class GuassianBlur:
    """Gaussian blur for smoothing"""
    def __init__(self, r, sigma=1):
        self.r = r
        self.kernel = np.empty(2 * r + 1)
        total = 0
        for i in range(2 * r + 1):
            self.kernel[i] = np.exp(-((i - r) ** 2) / (2 * sigma ** 2)) / ((2 * pi)**1/2 * sigma ** 2)
            total += self.kernel[i]
        self.kernel /= total

    def guassian_blur(self, mesh, flag=0):
        b, l, k = mesh.shape
        mesh_copy = np.zeros([b + 2 * self.r, l, k])
        mesh_result = np.zeros([b + 2 * self.r, l, k])
        mesh_copy[:self.r, :, :] = mesh[0, :, :]
        mesh_copy[self.r:b + self.r, :, :] = mesh
        mesh_copy[b + self.r:b + 2 * self.r, :, :] = mesh[-1, :, :]

        for i in range(k):
            for j in range(self.r, self.r + b):
                mesh_result[j, 0, i] = np.sum(self.kernel * mesh_copy[j - self.r: j + self.r + 1, 0, i])

        return mesh_result[self.r:self.r + b, :, :]


class MLP(torch.nn.Module):
    """MLP model for sign connector"""
    def __init__(self, input_dim=322):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SignConnectorWrapper:
    """Wrapper for sign connector to generate smooth transitions between glosses"""
    
    def __init__(self, connector_path, smplx_data_path, vn_dictionary_path, 
                 smplx_model_folder, device='cuda'):
        """
        Initialize Sign Connector
        
        Args:
            connector_path: Path to connector.pth
            smplx_data_path: Path to smplx_all.pkl
            vn_dictionary_path: Path to vn_dictionary.pkl
            smplx_model_folder: Path to SMPL-X model folder (REQUIRED)
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32
        
        # Joint indices
        self.joint_idx = np.array([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], dtype=np.int32)
        
        # Load connector model
        input_dim = len(self.joint_idx) * 3 * 2 + len(self.joint_idx)  # 322
        self.connector = MLP(input_dim=input_dim)
        self.connector.load_state_dict(torch.load(connector_path, map_location=self.device), strict=True)
        self.connector.to(self.device)
        self.connector.eval()
        
        # Load SMPL-X model - REQUIRED
        if smplx_model_folder is None:
            raise ValueError("smplx_model_folder is required to compute interpolation length")
        
        # Prepare model parameters
        # NOTE: We provide all parameters from data, so create_* should be False
        # use_pca=False is important: we don't use PCA compression for hand poses
        # model_type='smplx' is CRITICAL to load SMPL-X instead of SMPL
        model_params = dict(
            model_path=smplx_model_folder,
            model_type='smplx',  # CRITICAL: Must specify smplx to load correct model
            create_global_orient=False,  
            create_body_pose=False,  
            create_betas=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_expression=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_transl=False,
            use_pca=False,  # CRITICAL: Don't use PCA for hand poses
            dtype=dtype
        )
        
        # Add JointMapper if utils available (for compatibility with original code)
        if UTILS_AVAILABLE:
            mapping = smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                        use_face_contour=False, openpose_format='coco25')
            joint_mapper = JointMapper(mapping)
            model_params['joint_mapper'] = joint_mapper
        
        self.smplx_model = smplx.create(**model_params)
        self.smplx_model = self.smplx_model.to(device=self.device)
        self.smplx_model.eval()
        print("SMPL-X model loaded successfully")
        
        # Load SMPL-X data
        print(f"Loading SMPL-X data from {smplx_data_path}")
        with open(smplx_data_path, 'rb') as f:
            self.smplx_data = pickle.load(f)
        
        # Load dictionary
        print(f"Loading Vietnamese dictionary from {vn_dictionary_path}")
        with open(vn_dictionary_path, 'rb') as f:
            self.vn_dictionary = pickle.load(f)
        
        print("Sign Connector initialized successfully")
    
    def _compute_joints_location(self, est_params):
        """Compute joints location from SMPL-X model"""
        # Convert to tensor
        tensor_params = {}
        dtype = torch.float32
        
        # Keys that should NOT be passed to SMPL-X model
        skip_keys = ['camera_rotation', 'camera_translation']
        
        for key, val in est_params.items():
            if key in skip_keys:
                continue
            
            tensor_val = torch.from_numpy(np.asarray(val)).to(device=self.device, dtype=dtype)
            if tensor_val.ndim == 1:
                tensor_val = tensor_val.unsqueeze(0)
            tensor_params[key] = tensor_val
        
        # Forward pass
        with torch.no_grad():
            model_output = self.smplx_model(**tensor_params)
            joints_location = model_output.joints
        
        # Select joints
        joints_idx_tensor = torch.tensor(self.joint_idx, dtype=torch.long).to(device=self.device)
        joints_location = torch.index_select(joints_location, 1, joints_idx_tensor)
        
        return joints_location
    
    def _compute_interpolation_length(self, data_0, data_1):
        """Compute interpolation length using connector MLP - ALWAYS uses model output"""
        # Compute joints locations
        joints_location_pre = self._compute_joints_location(data_0)
        joints_location_nex = self._compute_joints_location(data_1)
        
        # Compute joints distance
        joints_dis = torch.sqrt(((joints_location_pre - joints_location_nex) ** 2).sum(dim=-1))
        
        # Reshape
        joints_location_pre = joints_location_pre.reshape([1, -1])
        joints_location_nex = joints_location_nex.reshape([1, -1])
        
        # Forward through connector
        with torch.no_grad():
            input_tensor = torch.cat((joints_location_pre, joints_location_nex, joints_dis), 1)
            len_inter = self.connector(input_tensor)
            len_inter = max(round(len_inter.item()), 1)
        
        return len_inter
    
    def process_glosses(self, glosses, output_dir):
        """
        Process glosses and generate motion files with smooth transitions
        
        Args:
            glosses: List of gloss strings or space-separated gloss string
            output_dir: Directory to save motion PKL files
            
        Returns:
            num_frames: Number of frames generated
        """
        if isinstance(glosses, str):
            glosses = glosses.split()
        
        # Filter valid glosses
        valid_glosses = []
        for gloss in glosses:
            if gloss in self.vn_dictionary and gloss in self.smplx_data:
                valid_glosses.append(gloss)
            else:
                print(f"Warning: Gloss '{gloss}' not found, skipping")
        
        if not valid_glosses:
            raise ValueError("No valid glosses found")
        
        print(f"Processing {len(valid_glosses)} glosses: {valid_glosses}")
        
        # Create motion sequence
        os.makedirs(output_dir, exist_ok=True)
        
        est_params_all = []
        inter_flag = []
        
        for gloss_idx, gloss in enumerate(valid_glosses):
            render_results = self.smplx_data[gloss]
            
            # Add all frames from this gloss
            for frame_data in render_results:
                est_params = {}
                for key, val in frame_data.items():
                    est_params[key] = val
                est_params_all.append(est_params)
                inter_flag.append(False)
            
            # Interpolation between glosses
            if gloss_idx < len(valid_glosses) - 1:
                next_gloss = valid_glosses[gloss_idx + 1]
                
                if next_gloss not in self.smplx_data:
                    print(f"Warning: next gloss {next_gloss} not found, skipping interpolation")
                    continue
                
                print(f"Interpolating between {gloss} and {next_gloss}")
                
                # Get last frame of current and first frame of next
                data_0 = render_results[-1]
                data_1 = self.smplx_data[next_gloss][0]
                
                # Compute interpolation length using connector - ALWAYS uses model
                len_inter = self._compute_interpolation_length(data_0, data_1)
                print(f"Interpolation length: {len_inter}")
                
                # Calculate weights
                weights = np.zeros(len_inter)
                interval = 1.0 / (len_inter + 1)
                for i in range(len_inter):
                    weights[i] = 1.0 - (i + 1) * interval
                
                # Create interpolated frames
                for idx_w, weight in enumerate(weights):
                    est_params = {}
                    for key, val in data_0.items():
                        if key in ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression']:
                            est_params[key] = weight * data_0[key] + (1 - weight) * data_1[key]
                        else:
                            est_params[key] = data_0[key]
                    
                    est_params_all.append(est_params)
                    inter_flag.append(True)
        
        # Apply smoothing
        self._apply_smoothing(est_params_all)
        
        # Save motion files
        for i, est_params in enumerate(est_params_all):
            # Zero out certain body pose components
            if 'body_pose' in est_params:
                est_params['body_pose'][:, 0:15] = 0.
                est_params['body_pose'][:, 18:24] = 0.
                est_params['body_pose'][:, 27:33] = 0.
            
            fname = os.path.join(output_dir, f"{i:03d}.pkl")
            if inter_flag[i]:
                fname = os.path.join(output_dir, f"{i:03d}_inter.pkl")
            
            with open(fname, 'wb') as f:
                pickle.dump(est_params, f)
        
        print(f"Generated {len(est_params_all)} frames in {output_dir}")
        return len(est_params_all)
    
    def _apply_smoothing(self, est_params_all):
        """Apply Gaussian smoothing to motion parameters"""
        if len(est_params_all) == 0:
            return
        
        print(f"Total frames: {len(est_params_all)}")
        print(f"Available keys: {list(est_params_all[0].keys())}")
        print("Starting smoothing process...")
        
        for key, val in est_params_all[0].items():
            print(f"Processing key: {key}")
            try:
                if key == 'camera_rotation':
                    date_temp = np.zeros([len(est_params_all), 1, 9])
                    for i in range(len(est_params_all)):
                        date_temp[i] = est_params_all[i][key].reshape(1, 9)
                    GuassianBlur_ = GuassianBlur(1)
                    out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                    for i in range(len(est_params_all)):
                        est_params_all[i][key] = out_smooth[i].reshape(1, 3, 3)
                elif key == 'betas':
                    for i in range(len(est_params_all)):
                        est_params_all[i][key] = np.asarray([[0.421, -1.658, 0.361, 0.314, 0.226,
                                                                0.065, 0.175, -0.150, -0.097, -0.191]])
                elif key == 'global_orient':
                    for i in range(len(est_params_all)):
                        est_params_all[i][key] = np.asarray([[0, 0, 0]])
                else:
                    val_shape = est_params_all[0][key].shape
                    if len(val_shape) >= 2 and val_shape[1] > 0:
                        date_temp = np.zeros([len(est_params_all), 1, val_shape[1]])
                        for i in range(len(est_params_all)):
                            date_temp[i] = est_params_all[i][key]
                        GuassianBlur_ = GuassianBlur(1)
                        out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                        for i in range(len(est_params_all)):
                            est_params_all[i][key] = out_smooth[i]
                    else:
                        print(f"Skipping smoothing for {key}: shape {val_shape}")
            except Exception as e:
                print(f"Error processing {key}: {e}")