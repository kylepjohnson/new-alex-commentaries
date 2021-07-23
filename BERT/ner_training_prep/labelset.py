"""This module labels the training data for ner citation classification training."""

import itertools
from typing import List

from .alignment import align_tokens_and_annotations_bilou


class LabelSet:
    """This class assigns labels for input ner training data."""

    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, bilu) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            new_label = f"{bilu}-{label}"
            self.labels_to_id[new_label] = num
            self.ids_to_label[num] = new_label
        # Add the OUTSIDE label - no label for the token

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
        return list(map(self.labels_to_id.get, raw_labels))
