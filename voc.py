"""
    voc.py
    Oct 15 2022
    Gabriel Moreira
"""

import torch
from collections import Counter, OrderedDict

class Vocabulary:
    def __init__(self, words):
        """
            Creates vocabulary from list of words
        """
        words_counter = Counter(words)
        words_sorted  = sorted(words_counter.items(), key=lambda x : x[1], reverse=True)
        words_dict    = OrderedDict(words_sorted)
        
        self._word2idx = {key : i for i, key in enumerate(words_dict.keys())}
        self._idx2word = {i : key for i, key in enumerate(words_dict.keys())}

        self.length = len(self._word2idx)

        
    def __len__(self):
        return self.length

    
    def w2i(self, src):
        """
            Word (or iterable of words) to index (or list of indices)
        """
        if isinstance(src, str):
            return self._word2idx[src]
        else:
            out = []
            for word in src:
                out.append(self._word2idx[word])
            return out

        
    def i2w(self, src):
        """
            Index (or iterable of indicex) to word (or list of words)
        """
        if isinstance(src, int):
            return self._idx2word[src]
        else:
            out = []
            if isinstance(src, torch.Tensor):
                for idx in src.ravel():
                    out.append(self._idx2word[idx.item()])
            else:
                for idx in src:
                    out.append(self._idx2word[idx])     
            return out
