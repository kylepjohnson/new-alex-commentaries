#!/usr/bin/env python
# coding: utf-8

# In[21]:


import re
import torch
import json
import pickle

from seqeval.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from seqeval.scheme import BILOU
from transformers import BertForTokenClassification, BertTokenizer, BertTokenizerFast, BatchEncoding
from tokenizers import decoders, Encoding

from ner_pipeline.labelset import LabelSet


# In[4]:


model = BertForTokenClassification.from_pretrained("bert_ner_il_od-with-gpu-10000.model")


# In[5]:


tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


# In[10]:


# load the object
with open("labeled_data/test_1500_of_10000.pickle", "rb") as dataset_test_file:
    test_instances = pickle.load(dataset_test_file)


# In[11]:


#print(test_instances[0])


# In[25]:


pred = []
for instance in test_instances:
    inputs_line = tokenizer.encode(instance["content"], return_tensors="pt")
    outputs_line = model(inputs_line).logits
    predictions_line = torch.argmax(outputs_line, dim=2)
    
    pred_line_label = []
    for prediction in predictions_line[0].numpy():
        pred_line_label.append(label_list[prediction])
    pred.append(pred_line_label)


# In[26]:


citation_label_set = LabelSet(labels=["Citation"])


# In[27]:


label_list = ['O', 'B-Citation', 'I-Citation', 'L-Citation', 'U-Citation']


# In[28]:


true = []
for instance in test_instances:
    match_tokenized_batch : BatchEncoding = tokenizer(instance["content"])
    match_tokenized_text : Encoding = match_tokenized_batch[0]
    aligned_label_ids = clentity_label_set.get_aligned_label_ids_from_annotations(
        match_tokenized_text, instance["annotations"]
    )
    true_line_label = []
    for match_id in aligned_label_ids:
        true_line_label.append(label_list[match_id])
    true.append(true_line_label)


# In[29]:


print("Length of predictions: " + str(len(pred)))
print("Length of truths: " + str(len(true)))


# In[30]:


print(classification_report(true, pred, mode='strict'))


# In[ ]:




