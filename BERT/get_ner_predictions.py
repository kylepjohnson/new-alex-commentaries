import re
import torch
import json

from transformers import BertForTokenClassification, BertTokenizer, BertTokenizerFast, BatchEncoding
from tokenizers import decoders, Encoding
from alignment import align_tokens_and_annotations_bilou

from prepareData import prepare_data

def get_predictions(filename, outputName):
    label_list = ['O', 'B-CLEntity', 'I-CLEntity', 'L-CLEntity', 'U-CLEntity']
    model = BertForTokenClassification.from_pretrained("bert_ner_finetuned_iliad-with-gpu-pattern2.model")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    book_lines = []
    book = open(filename)
    for line in book:
        line = line.strip()
        book_lines.append(line)
    book_lines = [line for line in book_lines if line]

    pred = []
    for line in book_lines[:10]:
        line_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(line)))
        line_inputs = tokenizer.encode(line, return_tensors="pt")
        line_outputs = model(line_inputs).logits
        line_predictions = torch.argmax(line_outputs, dim=2)
        line_pred_labels = []
        for prediction in line_predictions[0].numpy():
            line_pred_labels.append(label_list[prediction])
        pred.append(line_pred_labels)
        
    with open(outputName, 'w') as f:
        f.write(json.dumps(pred))
    return

if __name__ == "__main__":
    filename = "../example-texts/odyssey.txt"
    outputName = "odyssey_ner_predictions.txt"
    get_predictions(filename, outputName)