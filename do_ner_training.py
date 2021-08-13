#!/usr/bin/env python
# coding: utf-8


import requests
from bs4 import BeautifulSoup
from internetarchive import search_items, get_item, Search

from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW, BertTokenizer, BertTokenizerFast, BatchEncoding, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import List


# In[4]:


from ner_pipeline.scrape_archive import do_search, prepare_data
from ner_pipeline.containers import TraingingBatch
from ner_pipeline.dataset_ner import TrainingDataset
from ner_pipeline.labelset import LabelSet


# In[5]:


# Do search
il_od: str = "iliad OR odyssey AND mediatype:texts"  # 771,646 with full_text_search, 6240 without
search_res: Search = do_search(keyword_string=il_od)


# In[6]:


pattern = r'Iliad\s\d{1,2}\.\d{1,4}|Il\.*\s\d{1,2}\.\d{1,4}|Iliad\s.[ivxlcdm]*\.\s*\d{1,4}|             Il\.*\s.[ivxlcdm]*\.\s*\d{1,4}|book\s*.[ivxlcdm]\.\sline\s*\d{1,4}|             Odyssey\s\d{1,2}\.\d{1,4}|Od\.*\s\d{1,2}\.\d{1,4}|Odyssey\s.[ivxlcdm]*\.\s*\d{1,4}|             Od\.*\s.[ivxlcdm]*\.\s*\d{1,4}'


# In[7]:


labeled_data = prepare_data(search_res, pattern, num_of_pos = 10000, num_of_neg = 10000)


# In[8]:


print(len(labeled_data))


# In[9]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_set = LabelSet(labels=["Citation"]) #Only one label in this dataset


# In[10]:


il_od_ner_trainingData = TrainingDataset(
    data=labeled_data, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)


# In[11]:


dataset_train = il_od_ner_trainingData[:len(il_od_ner_trainingData) // 2]
dataset_eval = il_od_ner_trainingData[len(il_od_ner_trainingData) // 2:]


# In[12]:


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(il_od_ner_trainingData.label_set.ids_to_label.values())
)


# In[13]:


training_args = TrainingArguments("test_trainer")


# In[14]:


trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset_train, eval_dataset=dataset_eval
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model('bert_ner_il_od-with-gpu-10000.model')

