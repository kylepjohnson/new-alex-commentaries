"""This module builds up the ner training dataset for ner citation classification training."""

from dataclasses import dataclass
from typing import List, Any
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerFast

from .labelset import LabelSet

IntList = List[int]  # A list of token_ids
IntListList = List[IntList]  # A List of List of token_ids, e.g. a Batch


@dataclass
class TrainingExample:
    """This class shows the data structure inherited by TrainingDataset."""

    input_ids: IntList
    attention_mask: IntList
    labels: IntList


class TrainingDataset(Dataset):
    """This class builds up ner Training dataset through tokenization, batching, and padding."""

    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        self.texts = []
        self.annotations = []

        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for idx in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[idx]
            raw_annotations = self.annotations[idx]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        # empty_label_id = "O" no matching label
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):

                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        labels=(
                            label[start:end]
                            + [-100] * padding_to_add  # padding if needed
                        ),  # -100 is a special token for padding of labels,
                        attention_mask=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attenetion masks where we added padding
                        ),
                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:
        return self.training_examples[idx]
