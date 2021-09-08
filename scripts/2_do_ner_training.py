#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # About
# 
# This notebook takes the training data in `pos_neg_instances` and outputs a model.

# # Load and preprocess data

# In[2]:


import ast
from dataclasses import dataclass
import json
import pickle
import random
from typing import List, Union

from bs4 import BeautifulSoup
import requests
from internetarchive import search_items, get_item, Search
from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW, BertTokenizer, BertTokenizerFast, BatchEncoding, TrainingArguments, Trainer

from ner_pipeline.scrape_for_training import do_search, prepare_data, load_scraped_data, get_scraped_dataset_size
from ner_pipeline.containers import TraingingBatch
from ner_pipeline.dataset_ner import TrainingDataset
from ner_pipeline.dataset_ner import TrainingExample
from ner_pipeline.labelset import LabelSet


# In[3]:


# Directly load the previously scraped pos/neg instances from saved text files.
NUM_EXAMPLES: int = 10  # 10000
POS_INSTANCES: list[dict[str, Union[str, dict[str, Union, str, int]]]] =     load_scraped_data(f"pos_neg_instances/pos_instances_{NUM_EXAMPLES}.txt")
NEG_INSTANCES: list[dict[str, Union[str, dict[str, Union, str, int]]]] =     load_scraped_data(f"pos_neg_instances/neg_instances_{NUM_EXAMPLES}.txt")


# In[4]:


# Since did not find 10000 positive instances and 10000 negative instances, 
# we take 5000 each in this case.
NUM_FOR_TRAINING: int = 10  # 5000
LABELED_DATA: list[dict[str, Union[str, dict[str, Union, str, int]]]] =     POS_INSTANCES[:NUM_FOR_TRAINING] + NEG_INSTANCES[:NUM_FOR_TRAINING]


# In[5]:


len(LABELED_DATA)


# In[6]:


print(LABELED_DATA[:3])


# In[7]:


# note: in future, consider shuffling pos/neg seperately
random.shuffle(LABELED_DATA)


# In[8]:


DATASET_SIZE: int = len(LABELED_DATA)
print(DATASET_SIZE)


# In[9]:


DATASET_TRAIN: list[dict[str, Union[str, dict[str, Union, str, int]]]] =     LABELED_DATA[:DATASET_SIZE*17//20] # 85% training data
DATASET_TRAIN_SIZE = len(DATASET_TRAIN)
print(DATASET_TRAIN_SIZE)


# In[10]:


DATASET_TEST = LABELED_DATA[DATASET_SIZE*17//20:] # 15% testing data
DATASET_TEST_SIZE = len(DATASET_TEST)

print("Number of instances for training: " + str(DATASET_TRAIN_SIZE))
print("Number of instances for testing: " + str(DATASET_TEST_SIZE))


# In[11]:


# Save instances for training(train and eval) and testing
TRAIN_FP: str = f"labeled_data/train_{DATASET_TRAIN_SIZE}_of_{DATASET_SIZE}.pickle"
TEST_FP: str = f"labeled_data/test_{DATASET_TEST_SIZE}_of_{DATASET_SIZE}.pickle"

with open(TRAIN_FP, "wb") as DATASET_TRAIN_FILE:
    pickle.dump(DATASET_TRAIN, DATASET_TRAIN_FILE)

with open(TEST_FP, "wb") as DATASET_TEST_FILE:
    pickle.dump(DATASET_TEST, DATASET_TEST_FILE)


# In[12]:


# load the pickle file
LOAD_TRAINING_PICKLES = False
if LOAD_TRAINING_PICKLES:
    with open(TRAIN_FP, "rb") as DATASET_MODEL_FILE:
        DATASET_TRAIN = pickle.load(DATASET_MODEL_FILE)
    with open(TEST_FP, "rb") as DATASET_MODEL_FILE:
        DATASET_TEST = pickle.load(DATASET_MODEL_FILE)


# # Configure BERT for training

# In[13]:


TOKENIZER: BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-cased')


# In[14]:


LABEL_SET: LabelSet = LabelSet(labels=["Citation"]) #Only one label in this dataset


# In[15]:


# TODO: Understand why more results are returned than sent

IL_OD_TRAINING_DATASET: TrainingDataset = TrainingDataset(
    data=DATASET_TRAIN, tokenizer=TOKENIZER, label_set=LABEL_SET, tokens_per_batch=16
)
print(len(IL_OD_TRAINING_DATASET))


# In[16]:


print(IL_OD_TRAINING_DATASET[3])


# In[17]:


IL_OD_NER_TRAIN: list[TrainingExample] = IL_OD_TRAINING_DATASET[:len(IL_OD_TRAINING_DATASET)*17//20]
IL_OD_NER_EVAL = IL_OD_TRAINING_DATASET[len(IL_OD_TRAINING_DATASET)*17//20:]
print("Size of dataset for train: " + str(len(IL_OD_NER_TRAIN)))
print("Size of dataset for eval: " + str(len(IL_OD_NER_EVAL)))


# In[18]:


# Get the label list
print(IL_OD_TRAINING_DATASET.label_set.ids_to_label.values())


# In[19]:


MODEL: BertForTokenClassification = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(IL_OD_TRAINING_DATASET.label_set.ids_to_label.values())
)


# In[20]:


TRAINING_ARGS: TrainingArguments = TrainingArguments("test_trainer")


# In[21]:


TRAINER: Trainer = Trainer(
    model=MODEL,
    args=TRAINING_ARGS,
    train_dataset=IL_OD_NER_TRAIN,
    eval_dataset=IL_OD_NER_EVAL
)


# In[23]:


TRAINER.train()


# In[24]:


TRAINER.save_model(f"bert_ner_il_od-with-gpu-{NUM_EXAMPLES}.model")


# In[ ]:




