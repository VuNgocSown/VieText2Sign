"""Configuration for Gloss Retrieval"""


class RetrievalConfig:
    """Configuration for retrieval system"""
    
    def __init__(
        self,
        gloss_dictionary_path='./data/gloss_dictionary.pkl',
        protonx_api_key=None,
        max_ngram=5,
        min_ngram=1,
        embedding_threshold=0.95,
        device='cpu'
    ):
        self.gloss_dictionary_path = gloss_dictionary_path
        self.protonx_api_key = protonx_api_key
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.embedding_threshold = embedding_threshold
        self.device = device
