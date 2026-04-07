"""Gloss Retriever with Underthesea tokenization"""
import pickle
import numpy as np
import os
from typing import List, Dict, Optional

from .config import RetrievalConfig
from .utils import find_non_overlapping_matches

try:
    from underthesea import word_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    print("Warning: underthesea not installed. Install with: pip install underthesea")


class GlossRetriever:
    """Gloss retriever with Underthesea word segmentation"""
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config if config else RetrievalConfig()
        
        if not UNDERTHESEA_AVAILABLE:
            raise ImportError(
                "underthesea is required for this retriever. "
                "Install with: pip install underthesea"
            )
        
        # Load unified database
        db_file = 'data/gloss_db.pkl'
        print(f"Loading gloss database from {db_file}")
        with open(db_file, 'rb') as f:
            self.db = pickle.load(f)
        
        self.glosses = list(self.db.keys())
        self.gloss_set = set(self.glosses)
        
        # Extract embeddings for fast search
        self.embeddings = np.array([self.db[g]['embedding'] for g in self.glosses])
        
        # Init ProtonX for query embedding
        from protonx import ProtonX
        api_key = self.config.protonx_api_key or os.getenv('PROTONX_API_KEY')
        if not api_key:
            raise ValueError("ProtonX API key required")
        self.client = ProtonX(api_key=api_key)
        
        print(f"Loaded {len(self.glosses)} glosses")
        print(f"  With SMPLX: {sum(1 for v in self.db.values() if v['smplx'] is not None)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text using Underthesea
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            List of tokens in uppercase, with underscores for compound words
            
        Example:
            "Tôi học tiếng Anh" -> ['TÔI', 'HỌC', 'TIẾNG_ANH']
        """
        # Use underthesea for word segmentation
        tokens = word_tokenize(text)
        
        # Convert to uppercase and replace spaces with underscores in compound words
        processed_tokens = []
        for token in tokens:
            # If token contains space (compound word), replace with underscore
            token_upper = token.upper().replace(' ', '_')
            processed_tokens.append(token_upper)
        
        return processed_tokens
    
    def longest_match(self, tokens: List[str]):
        """Longest match from dictionary"""
        matches = []
        for n in range(self.config.max_ngram, self.config.min_ngram - 1, -1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i+n])
                if ngram in self.gloss_set:
                    matches.append((ngram, i, i + n))
        return find_non_overlapping_matches(matches)
    
    def generate_ngrams(self, tokens: List[str]) -> List[tuple]:
        """Generate n-grams from tokens"""
        ngrams = []
        for n in range(self.config.min_ngram, min(self.config.max_ngram, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                ngram_text = '_'.join(tokens[i:i+n])
                ngrams.append((ngram_text, i, i + n))
        return ngrams
    
    def embedding_match(self, tokens: List[str], matched_positions: set):
        """Embedding similarity search"""
        matches = []
        ngrams = self.generate_ngrams(tokens)
        
        for ngram_text, start, end in ngrams:
            if set(range(start, end)).intersection(matched_positions):
                continue
            if ngram_text in self.gloss_set:
                continue
            
            # Query embedding
            query_text = ngram_text.replace('_', ' ')
            emb = self.client.embeddings.create(query_text)
            if isinstance(emb, dict):
                emb = emb.get('embedding', emb.get('data', emb))
            query_emb = np.array(emb, dtype=np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)
            
            # Find best match
            similarities = np.dot(self.embeddings, query_emb)
            best_idx = np.argmax(similarities)
            best_sim = float(similarities[best_idx])
            
            if best_sim >= self.config.embedding_threshold:
                best_gloss = self.glosses[best_idx]
                matches.append((best_gloss, start, end, best_sim))
        
        return matches
    
    def retrieve(self, text: str) -> Dict:
        """
        Retrieve glosses with SMPLX data using Underthesea tokenization
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            {
                'input': str,
                'output': str,  # Space-separated glosses
                'glosses': [
                    {
                        'name': 'GLOSS',
                        'type': 'exact' or 'embedding',
                        'similarity': float,
                        'smplx': [...] or None,
                        'num_frames': int or 0
                    },
                    ...
                ]
            }
        """
        # Tokenize with Underthesea
        tokens = self.tokenize(text)
        if not tokens:
            return {'input': text, 'output': '', 'glosses': []}
        
        # Longest match
        exact_matches = self.longest_match(tokens)
        matched_positions = set()
        for _, start, end in exact_matches:
            matched_positions.update(range(start, end))
        
        # Embedding match
        emb_matches = self.embedding_match(tokens, matched_positions)
        
        # Build result
        glosses_result = []
        
        for gloss, start, end in exact_matches:
            info = self.db[gloss]
            glosses_result.append({
                'name': gloss,
                'start': start,
                'end': end,
                'type': 'exact',
                'similarity': 1.0,
                'smplx': info['smplx'],
                'num_frames': len(info['smplx']) if info['smplx'] else 0
            })
        
        for gloss, start, end, sim in emb_matches:
            info = self.db[gloss]
            glosses_result.append({
                'name': gloss,
                'start': start,
                'end': end,
                'type': 'embedding',
                'similarity': sim,
                'smplx': info['smplx'],
                'num_frames': len(info['smplx']) if info['smplx'] else 0
            })
        
        # Sort by position
        glosses_result.sort(key=lambda x: x['start'])
        
        output = ' '.join([g['name'] for g in glosses_result])
        
        return {
            'input': text,
            'output': output,
            'glosses': glosses_result
        }
