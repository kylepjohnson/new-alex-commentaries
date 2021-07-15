#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

from typing_extensions import TypedDict
from typing import List,Any
import re


# In[2]:


IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch


# In[3]:


pattern = r' ([A-Z].[a-z]+)'
re.compile(pattern)

def get_annotations(text, pattern):
    annotations = []
    for match in re.finditer(pattern, text):
        label_dic = dict()
        label_dic['start'] = match.start()
        label_dic['end'] = match.end()
        label_dic['label'] = 'CLEntity' # Entity starting with a capital letter
        annotations.append(label_dic)
    return annotations


# In[4]:


json_data = []
book = open("../example-texts/iliad.txt")
for line in book:
    line = line.strip()
    
    line_data = dict()
    line_data['content'] = line
    line_data['annotations'] = get_annotations(line, pattern)
    json_data.append(line_data)


# In[5]:


# with open('example-texts/iliad.txt') as fo:
#     text = fo.read()


# In[6]:


# annotations = []
# for match in re.finditer(pattern, text):
#     label_dic = dict()
#     label_dic['start'] = match.start()
#     label_dic['end'] = match.end()
#     label_dic['text'] = text[match.start():match.end()]
#     label_dic['label'] = 'CL-Entity' # Entity starting with a capital letter
#     annotations.append(label_dic)
# print(len(annotations))


# In[7]:


from transformers import BertTokenizerFast,  BatchEncoding
from tokenizers import Encoding

def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations):
    tokens = tokenized.tokens
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        annotation_token_ix_set = set()# A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)

        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
    return aligned_labels


# In[8]:


## try an exmaple
# example = {'content': 'We encourage people to read and share the Early Journal Content openly and to tell others that this', 'annotations': [{'start': 0, 'end': 2, 'label': 'CLEntity'}, {'start': 42, 'end': 63, 'label': 'CLEntity'}]}
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased') # Load a pre-trained tokenizer
# tokenized_batch : BatchEncoding = tokenizer(example["content"])
# tokenized_text : Encoding = tokenized_batch[0]
# labels = align_tokens_and_annotations_bilou(tokenized_text, example["annotations"])

# for token, label in zip(tokenized_text.tokens, labels):
#     print(token, "-", label)


# In[9]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


# In[10]:


import itertools

class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        # Add the OUTSIDE label - no label for the token

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
        return list(map(self.labels_to_id.get, raw_labels))


# example_label_set = LabelSet(labels=["CLEntity"])
# aligned_label_ids = example_label_set.get_aligned_label_ids_from_annotations(
#     tokenized_text, example["annotations"]
# )
# tokens = tokenized_text.tokens
# for token, label in zip(tokens, aligned_label_ids):
#     print(token, "-", label)


# In[11]:


from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


# In[12]:


@dataclass
class TrainingExample:
    input_ids: IntList
    attention_mask: IntList
    labels: IntList


class TrainingDataset(Dataset):
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
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        empty_label_id = "O"
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


# In[13]:


label_set = LabelSet(labels=["CLEntity"])
ds = TrainingDataset(
    data=json_data, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)


# In[14]:


len(ds) // 2


# In[15]:


dataset_train = ds[:len(ds) // 2]


# In[16]:


dataset_eval = ds[len(ds) // 2:]


# In[17]:


print(len(dataset_train))
print(len(dataset_eval))


# In[18]:


import torch


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


# In[19]:


from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW, BertTokenizer
import torch


# In[20]:


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(ds.label_set.ids_to_label.values())
)


# In[21]:


from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer")


# In[22]:


from transformers import Trainer

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset_train, eval_dataset=dataset_eval
)


# In[ ]:


trainer.train()


# In[ ]:




trainer.save_model('bert_ner_finetuned_iliad.model')
