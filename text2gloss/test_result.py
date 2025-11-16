"""Interactive prediction script for Text2Gloss"""
from . import config
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def predict(text, model, tokenizer):
    """Predict gloss for input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=config.MAX_LENGTH,
        truncation=True
    )
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    
    gloss = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gloss


def main():
    """Interactive prediction loop"""
    print("="*60)
    print("Text2Gloss - Interactive Prediction")
    print("="*60)
    
    model_path = f"{config.MODEL_OUTPUT_DIR}/best_model"
    
    # Load model and tokenizer
    print(f"\nLoading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(config.DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error: {e}")
        print("Please run train.py first to train the model.")
        return
    
    print("\nReady! Enter Vietnamese text to convert to Gloss.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            text = input("Input: ").strip()
            
            if text.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            gloss = predict(text, model, tokenizer)
            print(f"Gloss: {gloss}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
