import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        unique_words = sorted(list(set(" ".join(texts).lower().split(" "))))
        
        
        for i, word in enumerate(unique_words):
            self.word_to_id[word] = i + 4
            self.id_to_word[i + 4] = word
        
        self.vocab_size = len(self.word_to_id.keys())
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded = []
        preprocessed_text = text.lower().split(" ")
        for word in preprocessed_text:
            if word in self.word_to_id.keys():
                encoded.append(self.word_to_id[word])
            elif len(word) > 0:
                encoded.append(1)

        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = []
        for id in ids:
            if id < 4:
                continue 
            if id not in self.id_to_word.keys():
                decoded.append(self.unk_token)
                continue
            decoded.append(self.id_to_word[id])
        return " ".join(decoded)
