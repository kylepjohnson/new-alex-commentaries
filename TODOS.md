Tasks to do. Jiarong can mark as done
## Train
- Todo: Submit simple keyword search to archive.org API (eg, "iliad OR odyssey")
- Todo: Loop through the "hits". For each book, open up plain text and run the regex that matches eg Il. 7.100
- Todo: Write the positive examples to disk, for later training; also save (?) an equal number of negative examples
- Todo: Train model

## Inference
- Todo: Query archive api with the same keywords
- Todo: Loop through each book; but this time get the epub version, so can get page numbers
- Todo: Inference every (?) paragraph; if False then do nothing; if True, then write the result to a database (eg, .csv) and save (at least) book ID and page number (taken from the <span> tag

## Week 3
- Done: Read the epub version of the Seymour book (example in .ipynb)
- Done: Convert the output of the python epub library to something that BeautifulSoup can read
- Done: Loop through \<span\> tags and find the \<p\> tags w/in them
- Done: Apply the regex ^\d{1,3}\.\s to get the start of a comment on a line
- Done: Do something similar for the multi-line comments
- Done: Save to your data structure (alter it if it helps you!)

## Week 2
- Done: Parse Homer Commentary text file (.txt)
## Week 1
- Done: Make a Python module that has a minimum working version of how to connect to archive.org API; give a few examples using different search parameters
- Done: Read Archive.org API docs, make note in README.md
- Done: Do BeautifulSoup tutorial

