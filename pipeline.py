"""Main Text-to-Sign Language Video Pipeline"""
import os
import shutil
import json
from datetime import datetime


class Text2SignPipeline:
    """
    Complete pipeline: Vietnamese Text -> Gloss -> Sign Language Video
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Dictionary containing:
                - text2gloss_model_path: Path to trained text2gloss model
                - connector_path: Path to connector.pth
                - smplx_data_path: Path to smplx_all.pkl
                - vn_dictionary_path: Path to vn_dictionary.pkl
                - blender_path: Path to Blender executable
                - render_script_path: Path to render_avatar.py
                - temp_dir: Temporary directory for intermediate files
                - output_dir: Output directory for final videos
                - device: 'cuda' or 'cpu'
                - fps: Video frame rate (default: 30)
        """
        self.config = config
        self.temp_dir = config.get('temp_dir', './temp')
        self.output_dir = config.get('output_dir', './output')
        self.device = config.get('device', 'cuda')
        self.fps = config.get('fps', 30)
        
        # Initialize components
        # Lazy import to avoid circular dependency
        from text2gloss_predictor import Text2GlossPredictor
        from sign_connector_wrapper import SignConnectorWrapper
        from blender_renderer import BlenderRenderer
        from video_creator import VideoCreator
        
        print("="*60)
        print("Initializing Text-to-Sign Language Pipeline")
        print("="*60)
        
        print("\n[1/4] Loading Text2Gloss model...")
        self.text2gloss = Text2GlossPredictor(
            model_path=config['text2gloss_model_path'],
            device=self.device
        )
        
        print("\n[2/4] Loading Sign Connector...")
        self.sign_connector = SignConnectorWrapper(
            connector_path=config['connector_path'],
            smplx_data_path=config['smplx_data_path'],
            vn_dictionary_path=config['vn_dictionary_path'],
            smplx_model_folder=config['smplx_model_folder'],
            device=self.device
        )
        
        print("\n[3/4] Initializing Blender Renderer...")
        self.renderer = BlenderRenderer(
            blender_path=config.get('blender_path', 'blender'),
            render_script_path=config.get('render_script_path')
        )
        
        print("\n[4/4] Initializing Video Creator...")
        self.video_creator = VideoCreator(fps=self.fps)
        
        print("\n" + "="*60)
        print("Pipeline initialized successfully!")
        print("="*60)
    
    def process(self, text, video_id=None, cleanup=True):
        """
        Process Vietnamese text to sign language video
        
        Args:
            text: Vietnamese text string
            video_id: Optional video identifier (auto-generated if None)
            cleanup: Whether to clean up temporary files after processing
            
        Returns:
            result: Dictionary containing:
                - video_id: Video identifier
                - text: Input text
                - gloss: Predicted gloss
                - video_path: Path to output video
                - num_frames: Number of frames
                - success: True if successful
        """
        # Generate video ID if not provided
        if video_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = f"video_{timestamp}"
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_id}")
        print(f"{'='*60}")
        print(f"Input text: {text}")
        
        try:
            # Step 1: Text -> Gloss
            print(f"\n[Step 1/4] Predicting gloss...")
            gloss = self.text2gloss.predict(text)
            print(f"Predicted gloss: {gloss}")
            
            # Step 2: Gloss -> Motion files (with sign connector)
            print(f"\n[Step 2/4] Generating motion sequence...")
            motion_dir = os.path.join(self.temp_dir, video_id, 'motions')
            num_frames = self.sign_connector.process_glosses(gloss, motion_dir)
            print(f"Generated {num_frames} motion frames")
            
            # Step 3: Render motion -> Images
            print(f"\n[Step 3/4] Rendering with Blender...")
            image_dir = os.path.join(self.temp_dir, video_id, 'images')
            num_images = self.renderer.render(motion_dir, image_dir, video_id)
            print(f"Rendered {num_images} images")
            
            # Step 4: Images -> Video
            print(f"\n[Step 4/4] Creating video...")
            os.makedirs(self.output_dir, exist_ok=True)
            video_path = os.path.join(self.output_dir, f"{video_id}.mp4")
            self.video_creator.create_video(image_dir, video_path)
            
            # Cleanup temporary files
            if cleanup:
                print(f"\nCleaning up temporary files...")
                temp_video_dir = os.path.join(self.temp_dir, video_id)
                if os.path.exists(temp_video_dir):
                    shutil.rmtree(temp_video_dir)
            
            result = {
                'video_id': video_id,
                'text': text,
                'gloss': gloss,
                'video_path': video_path,
                'num_frames': num_images,
                'success': True
            }
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS!")
            print(f"{'='*60}")
            print(f"Video ID: {video_id}")
            print(f"Gloss: {gloss}")
            print(f"Frames: {num_images}")
            print(f"Video: {video_path}")
            print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: {e}")
            print(f"{'='*60}\n")
            
            return {
                'video_id': video_id,
                'text': text,
                'gloss': None,
                'video_path': None,
                'num_frames': 0,
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, texts, cleanup=True):
        """
        Process multiple texts
        
        Args:
            texts: List of Vietnamese text strings
            cleanup: Whether to clean up temporary files
            
        Returns:
            results: List of result dictionaries
        """
        results = []
        for i, text in enumerate(texts):
            video_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
            result = self.process(text, video_id=video_id, cleanup=cleanup)
            results.append(result)
        
        return results


def create_default_config():
    """Create default configuration template"""
    return {
        'text2gloss_model_path': './text2gloss/text2gloss_model/best_model',
        'connector_path': './data/connector.pth',
        'smplx_data_path': './data/smplx_all.pkl',
        'vn_dictionary_path': './data/vn_dictionary.pkl',
        'blender_path': 'blender',
        'render_script_path': './render_avatar.py',
        'temp_dir': './temp',
        'output_dir': './output',
        'device': 'cuda',
        'fps': 30
    }



