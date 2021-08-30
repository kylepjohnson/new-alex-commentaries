#!/usr/bin/env python
# coding: utf-8


import requests
import random
import pickle
import ast
import json
from bs4 import BeautifulSoup
from internetarchive import search_items, get_item, Search

from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW, BertTokenizer, BertTokenizerFast, BatchEncoding, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import List


# In[4]:


from ner_pipeline.scrape_for_training import do_search, prepare_data, load_scraped_data, get_scraped_dataset_size
from ner_pipeline.containers import TraingingBatch
from ner_pipeline.dataset_ner import TrainingDataset
from ner_pipeline.labelset import LabelSet


# In[5]:


# Do search on archive.org
il_od: str = "iliad OR odyssey AND mediatype:texts"  # 771,646 with full_text_search, 6240 without
search_res: Search = do_search(keyword_string=il_od)


# In[6]:


# Regex of patterns of citations
pattern = r'Iliad\s\d{1,2}\.\d{1,4}|Il\.*\s\d{1,2}\.\d{1,4}|Iliad\s.[ivxlcdm]*\.\s*\d{1,4}|             Il\.*\s.[ivxlcdm]*\.\s*\d{1,4}|book\s*.[ivxlcdm]\.\sline\s*\d{1,4}|             Odyssey\s\d{1,2}\.\d{1,4}|Od\.*\s\d{1,2}\.\d{1,4}|Odyssey\s.[ivxlcdm]*\.\s*\d{1,4}|             Od\.*\s.[ivxlcdm]*\.\s*\d{1,4}'


# In[7]:


# # By calling this fucntion, user-defined number of pos/neg instances will be saved in `pos_neg_instances` folder.
# # No need to redo the scraping part every time. 
# prepare_data(search_res, pattern, num_of_pos = 10000, num_of_neg = 10000)


# In[8]:


# # Directly load the previously scraped pos/neg instances from saved text files.
pos_instances = load_scraped_data("pos_neg_instances/pos_instances_10000.txt")
neg_instances = load_scraped_data("pos_neg_instances/neg_instances_10000.txt")


# In[9]:


# Since did not find 10000 positive instances and 10000 negative instances, we take 5000 each in this case.
labeled_data = pos_instances[:5000] + neg_instances[:5000]


# In[10]:


len(labeled_data)


# In[11]:


print(labeled_data[:3])


# In[12]:


# note: in future, consider shuffling pos/neg seperately
random.shuffle(labeled_data)


# In[13]:


dataset_size = len(labeled_data)

dataset_train = labeled_data[:dataset_size*17//20] # 85% training data
dataset_train_size = len(dataset_train)

dataset_test = labeled_data[dataset_size*17//20:] # 15% testing data
dataset_test_size = len(dataset_test)

print("Number of instances for training: " + str(dataset_train_size))
print("Number of instances for testing: " + str(dataset_test_size))


# In[14]:


# Save instances for training(train and eval) and testing
with open("labeled_data/train_" + str(dataset_train_size) + "_of_" + str(dataset_size) + ".pickle", "wb") as dataset_train_file:
    pickle.dump(dataset_train, dataset_train_file)

with open("labeled_data/test_" + str(dataset_test_size) + "_of_" + str(dataset_size) + ".pickle", "wb") as dataset_test_file:
    pickle.dump(dataset_test, dataset_test_file)


# In[15]:


# # load the pickle file
# with open("labeled_data/model_" + str(dataset_model_size) + "_of_" + str(dataset_size) + ".pickle", "rb") as dataset_model_file:
#     p2 = pickle.load(dataset_model_file)
# print(p2)


# In[16]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_set = LabelSet(labels=["Citation"]) #Only one label in this dataset


# In[17]:


# Note: todo understand why more results are returned than sent

il_od_ner_trainingData = TrainingDataset(
    data=dataset_train, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=16
)
# print(len(il_od_ner_trainingData))


# In[18]:


# print(il_od_ner_trainingData[100])


# In[19]:


il_od_ner_train = il_od_ner_trainingData[:len(il_od_ner_trainingData)*17//20]
il_od_ner_eval = il_od_ner_trainingData[len(il_od_ner_trainingData)*17//20:]
print("Size of dataset for train: " + str(len(il_od_ner_train)))
print("Size of dataset for eval: " + str(len(il_od_ner_eval)))


# In[20]:


# # Get the label list
# print(il_od_ner_trainingData.label_set.ids_to_label.values())


# In[21]:


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(il_od_ner_trainingData.label_set.ids_to_label.values())
)


# In[107]:


training_args = TrainingArguments("test_trainer")


# In[39]:


trainer = Trainer(
    model=model, args=training_args, train_dataset=il_od_ner_train, eval_dataset=il_od_ner_eval
)


# In[40]:


trainer.train()


# In[41]:


trainer.save_model('bert_ner_il_od-with-gpu-10000.model')


# In[ ]:




