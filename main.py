"""
Main script to run Text-to-Sign Pipeline
"""
import json
from pipeline import Text2SignPipeline


def main():
    # Load config
    print("Loading configuration...")
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize pipeline
    pipeline = Text2SignPipeline(config)
    
    # Process text
    text = "chung_toi an_uong com_rang"
    print(f"\n{'='*60}")
    print(f"Processing text: {text}")
    print(f"{'='*60}\n")
    
    result = pipeline.process(text, cleanup=False)
    
    if result['success']:
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Video ID: {result['video_id']}")
        print(f"Text: {result['text']}")
        print(f"Gloss: {result['gloss']}")
        print(f"Video: {result['video_path']}")
        print(f"Frames: {result['num_frames']}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("FAILED!")
        print(f"{'='*60}")
        print(f"Error: {result.get('error')}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

