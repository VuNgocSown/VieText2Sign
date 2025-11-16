"""Text to Gloss Prediction Module"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Text2GlossPredictor:
    """Predict Gloss from Vietnamese text using trained model"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained text2gloss model
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_length = 128
        
        print(f"Loading Text2Gloss model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try to load model, fallback to ignore safetensors if corrupted
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        except Exception as e:
            print(f"Warning: Error loading safetensors ({e}), trying with use_safetensors=False")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, 
                use_safetensors=False
            ).to(self.device)
        
        self.model.eval()
        print("Text2Gloss model loaded successfully")
    
    def predict(self, text):
        """
        Predict gloss from text
        
        Args:
            text: Vietnamese text string
            
        Returns:
            gloss: Predicted gloss string
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        gloss = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return gloss
    
    def predict_batch(self, texts):
        """
        Predict gloss for multiple texts
        
        Args:
            texts: List of Vietnamese text strings
            
        Returns:
            glosses: List of predicted gloss strings
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        glosses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return glosses

