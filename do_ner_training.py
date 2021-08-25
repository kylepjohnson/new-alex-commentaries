#!/usr/bin/env python
# coding: utf-8
import requests
import random
import pickle
from bs4 import BeautifulSoup
from internetarchive import search_items, get_item, Search

from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW, BertTokenizer, BertTokenizerFast, BatchEncoding, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import List


# In[4]:


from ner_pipeline.scrape_for_training import do_search, prepare_data
from ner_pipeline.containers import TraingingBatch
from ner_pipeline.dataset_ner import TrainingDataset
from ner_pipeline.labelset import LabelSet


# In[5]:


# Do search
il_od: str = "iliad OR odyssey AND mediatype:texts"  # 771,646 with full_text_search, 6240 without
search_res: Search = do_search(keyword_string=il_od)


# In[10]:


pattern = r'Iliad\s\d{1,2}\.\d{1,4}|Il\.*\s\d{1,2}\.\d{1,4}|Iliad\s.[ivxlcdm]*\.\s*\d{1,4}|             Il\.*\s.[ivxlcdm]*\.\s*\d{1,4}|book\s*.[ivxlcdm]\.\sline\s*\d{1,4}|             Odyssey\s\d{1,2}\.\d{1,4}|Od\.*\s\d{1,2}\.\d{1,4}|Odyssey\s.[ivxlcdm]*\.\s*\d{1,4}|             Od\.*\s.[ivxlcdm]*\.\s*\d{1,4}'


# In[20]:


labeled_data = prepare_data(search_res, pattern, num_of_pos = 10000, num_of_neg = 10000)


# In[21]:


print(len(labeled_data))


# In[27]:


random.shuffle(labeled_data)


# In[22]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_set = LabelSet(labels=["Citation"]) #Only one label in this dataset


# In[28]:


il_od_ner_trainingData = TrainingDataset(
    data=labeled_data, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)


# In[38]:


print(il_od_ner_trainingData[:3])


# In[48]:


dataset_size = len(il_od_ner_trainingData)
dataset_train = il_od_ner_trainingData[:dataset_size*7//10]
dataset_train_size = len(dataset_train)
dataset_eval = il_od_ner_trainingData[dataset_size*7//10:dataset_size*7//10+dataset_size*3//20]
dataset_eval_size = len(dataset_eval)
dataset_test = il_od_ner_trainingData[dataset_size*7//10+dataset_size*3//20:]
dataset_test_size = len(dataset_test)


# In[63]:


# Save train/eval/test
with open("labeled_data/train_" + str(dataset_train_size) + "_of_" + str(dataset_size) + ".pickle", "wb") as dataset_train_file:
    pickle.dump(dataset_train, dataset_train_file)

# # load the object
# with open("labeled_data/train_" + str(dataset_train_size) + "_of_" + str(dataset_size) + ".pickle", "rb") as dataset_train_file:
#     p2 = pickle.load(dataset_train_file)

with open("labeled_data/eval_" + str(dataset_eval_size) + "_of_" + str(dataset_size) + ".pickle", "wb") as dataset_eval_file:
    pickle.dump(dataset_eval, dataset_eval_file)

with open("labeled_data/test_" + str(dataset_test_size) + "_of_" + str(dataset_size) + ".pickle", "wb") as dataset_test_file:
    pickle.dump(dataset_test, dataset_test_file)


# In[64]:


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(il_od_ner_trainingData.label_set.ids_to_label.values())
)


# In[65]:


training_args = TrainingArguments("test_trainer")


# In[66]:


trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset_train, eval_dataset=dataset_eval
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model('bert_ner_il_od-with-gpu-10000.model')

