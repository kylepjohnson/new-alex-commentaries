#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import torch
import json
import pickle
from typing import Union

from seqeval.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from seqeval.scheme import BILOU
from transformers import BertForTokenClassification, BertTokenizer, BertTokenizerFast, BatchEncoding
from tokenizers import decoders, Encoding
from torch import Tensor
from ner_pipeline.labelset import LabelSet


# In[2]:


# load the model
NUM_EXAMPLES: int = 10  # 10000
MODEL: BertForTokenClassification = BertForTokenClassification.from_pretrained(f"bert_ner_il_od-with-gpu-{NUM_EXAMPLES}.model")


# In[3]:


LABEL_LIST: list[str] = ['O', 'B-Citation', 'I-Citation', 'L-Citation', 'U-Citation']


# In[4]:


TOKENIZER: BertTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-cased")


# In[5]:


# load the labeled data for testing
# this one works well: f"labeled_data/test_1500_of_10000.pickle"
DATASET_TEST_SIZE: int = 1500
DATASET_SIZE: int = 10000
TEST_FP: str = f"labeled_data/test_{DATASET_TEST_SIZE}_of_{DATASET_SIZE}.pickle"
with open(TEST_FP, "rb") as DATASET_TEST_FILE:
    TEST_INSTANCES: list[dict[str, Union[str, dict[str, Union, str, int]]]] =         pickle.load(DATASET_TEST_FILE)


# In[6]:


print(TEST_INSTANCES[0])


# In[7]:


# Get predictions from out trained model
PRED: list[list[str]] = list()
for INSTANCE in TEST_INSTANCES:
    INPUTS_LINE: Tensor = TOKENIZER.encode(INSTANCE["content"], return_tensors="pt")
    # predict by the model
    OUTPUTS_LINE: Tensor = MODEL(INPUTS_LINE).logits
    PREDICTIONS_LINE: Tensor = torch.argmax(OUTPUTS_LINE, dim=2)
    PRED_LINE_LABEL: list[str] = list()
    for PREDICTION in PREDICTIONS_LINE[0].numpy():
        PRED_LINE_LABEL.append(LABEL_LIST[PREDICTION])
    PRED.append(PRED_LINE_LABEL)


# In[8]:


CITATION_LABEL_SET: LabelSet = LabelSet(labels=["Citation"])


# In[10]:


# Get true results of citations to evaluate the performance of our model
TRUE_INSTANCES: list[list[str]] = list()
for INSTANCE in TEST_INSTANCES:
    MATCH_TOKENIZED_BATCH: BatchEncoding = TOKENIZER(INSTANCE["content"])
    MATCH_TOKENIZED_TEXT: Encoding = MATCH_TOKENIZED_BATCH[0]
    ALIGNED_LABEL_IDS: list[int] = CITATION_LABEL_SET.get_aligned_label_ids_from_annotations(
        MATCH_TOKENIZED_TEXT, INSTANCE["annotations"]
    )
    TRUE_LINE_LABEL: list[str] = list()
    for MATCH_ID in ALIGNED_LABEL_IDS:
        TRUE_LINE_LABEL.append(LABEL_LIST[MATCH_ID])
    TRUE_INSTANCES.append(TRUE_LINE_LABEL)


# In[11]:


print("Length of predictions: " + str(len(PRED)))
print("Length of truths: " + str(len(TRUE_INSTANCES)))


# In[12]:


# Precision and Recall result
print(classification_report(TRUE_INSTANCES, PRED, mode='strict'))


# In[ ]:




