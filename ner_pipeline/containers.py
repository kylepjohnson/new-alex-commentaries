"""This module defines a class for ner training batch."""

from dataclasses import dataclass
from typing import List
import torch


IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids, e.g. a Batch


@dataclass
class TrainingExample:
    """This class shows the data structure inherited by TrainingBatch."""

    input_ids: IntList
    attention_mask: IntList
    labels: IntList


class TraingingBatch:
    """This class defines Training Batch for ner citation classification training."""

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
        self.input_ids = torch.LongTensor(input_ids)  # pylint: disable=no-member
        self.attention_mask = torch.LongTensor(masks)  # pylint: disable=no-member
        self.labels = torch.LongTensor(labels)  # pylint: disable=no-member
