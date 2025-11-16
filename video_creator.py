"""Video Creator Module"""
import cv2
import os
import glob


class VideoCreator:
    """Create video from rendered images"""
    
    def __init__(self, fps=30):
        """
        Initialize Video Creator
        
        Args:
            fps: Frames per second for output video
        """
        self.fps = fps
    
    def create_video(self, image_dir, output_path):
        """
        Create video from images
        
        Args:
            image_dir: Directory containing PNG images
            output_path: Path to save output video
            
        Returns:
            success: True if video created successfully
        """
        # Get sorted list of images
        images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if not images:
            raise ValueError(f"No PNG images found in {image_dir}")
        
        print(f"Creating video from {len(images)} images")
        
        # Read first image to get dimensions
        first_img = cv2.imread(images[0])
        if first_img is None:
            raise ValueError(f"Failed to read first image: {images[0]}")
        
        height, width = first_img.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write images to video
        for i, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read {img_path}, skipping")
                continue
            
            video.write(img)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(images)} images")
        
        video.release()
        print(f"Video created: {output_path}")
        
        return True

