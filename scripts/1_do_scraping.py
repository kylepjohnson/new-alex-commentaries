#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # About
# 
# This notebook gives examples of how to query the Archive.prg API with a search string, then save the results locally in the format expected by the next notebook, `2_do_inference.ipynb`.

# In[2]:


from internetarchive import search_items, get_item, Search

from ner_pipeline.scrape_for_training import do_search
from ner_pipeline.scrape_for_training import prepare_data


# # Step 1: Define documents to query
# 
# This first step finds documents that *may* contain citations of Homer's *Iliad* or *Odyssey*.

# In[3]:


IL_OD: str = "iliad OR odyssey AND mediatype:texts"
# 543,608 results with full_text_search (as of 06 Sept 2021)
SEARCH_RES: Search = do_search(keyword_string=IL_OD)


# # Step 2: Query documents for citations
# 
# In the documents returned above, now look for citations that match our regex pattern.

# In[4]:


# Regex of patterns of citations
PATTERN = r'Iliad\s\d{1,2}\.\d{1,4}|Il\.*\s\d{1,2}\.\d{1,4}|Iliad\s.[ivxlcdm]*\.\s*\d{1,4}|             Il\.*\s.[ivxlcdm]*\.\s*\d{1,4}|book\s*.[ivxlcdm]\.\sline\s*\d{1,4}|             Odyssey\s\d{1,2}\.\d{1,4}|Od\.*\s\d{1,2}\.\d{1,4}|Odyssey\s.[ivxlcdm]*\.\s*\d{1,4}|             Od\.*\s.[ivxlcdm]*\.\s*\d{1,4}'


# By calling this fucntion, user-defined number of pos/neg instances will be saved in the directory `pos_neg_instances`.

# In[5]:


NUM_POS = 10  # 10000
NUM_NEG = 10  # 10000
prepare_data(search_res=SEARCH_RES,
             pattern=PATTERN,
             num_of_pos=NUM_POS,
             num_of_neg=NUM_NEG)


# # Inspect results
# 
# Now that the results have been downloaded, look at the two files that have been generated.

# In[6]:


get_ipython().system('head -n 10 pos_neg_instances/pos_instances_10.txt')


# In[7]:


get_ipython().system('head -n 10 pos_neg_instances/neg_instances_10.txt')


# # How to run
# 
# When scraping large amounts of samples from Archive.org, export this file and put it in the `scripts` directory.
