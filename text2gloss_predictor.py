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
        
        # Ấn định ngôn ngữ đích: chỉ cần với mBART/NLLB, None với mT5/MarianMT
        if hasattr(self.tokenizer, 'lang_code_to_id') and 'vi_VN' in self.tokenizer.lang_code_to_id:
            self.forced_bos_token_id = self.tokenizer.lang_code_to_id['vi_VN']
            self.model.config.forced_bos_token_id = self.forced_bos_token_id
        elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            # NLLB / token-based style
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids('vie_Latn')
                if tok_id != self.tokenizer.unk_token_id:
                    self.forced_bos_token_id = tok_id
                    self.model.config.forced_bos_token_id = tok_id
                else:
                    self.forced_bos_token_id = None
            except Exception:
                self.forced_bos_token_id = None
        else:
            self.forced_bos_token_id = None
        
        print(f"Text2Gloss model loaded (forced_bos_token_id={self.forced_bos_token_id})")
    
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
                early_stopping=True,
                forced_bos_token_id=self.forced_bos_token_id,
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
                early_stopping=True,
                forced_bos_token_id=self.forced_bos_token_id,
            )
        
        glosses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return glosses

