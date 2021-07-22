"""This module defines a class for training batch."""
import torch
from dataclasses import dataclass
from typing import List, Any

IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids, e.g. a Batch

"""This class shows the data structure inherited by TrainingBatch."""
@dataclass
class TrainingExample:
    input_ids: IntList
    attention_mask: IntList
    labels: IntList

"""This class defines Training Batch for ner citation classification training."""
class TraingingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_mask: torch.Tensor
        self.labels: torch.Tensor
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_mask)
            labels.append(ex.labels)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)
        