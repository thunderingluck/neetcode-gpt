import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        s = positive + negative
        words = set()
        for sentence in s:
            for word in sentence.split():
                words.add(word)
        vocab = sorted(words)

        word_to_id = {word : i + 1 for i, word in enumerate(vocab)}
        encoded = []

        for sentence in s:
            ids = []

            for word in sentence.split():
                ids.append(word_to_id[word])

            sentence_tensor = torch.tensor(ids)
            encoded.append(sentence_tensor)
        
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)