"""This module extracts the training data for ner citation classification training."""

import re


def get_annotations(text, pattern):
    """Helper function for prepare_data

    Args:
        text (str): Input string
        pattern (regex expression): pattern we are looking for from the input string

    Returns:
        list: A list of dictionaries recording matches we found.
        E.g. [{'start': int, 'end': int, 'label': str}, ]
    """
    annotations = []
    # find all strings matching the input pattern.
    for match in re.finditer(pattern, text):
        label_dic = dict()
        label_dic["start"] = match.start()
        label_dic["end"] = match.end()
        label_dic["label"] = "CLEntity"  # Entity starting with a capital letter.
        annotations.append(label_dic)
    return annotations


def prepare_data(filename, pattern):
    """Find all strings matching the input pattern within the text file.

    Args:
        filename (str): Path to a text file
        pattern (regex expression): pattern we are looking for

    Returns:
        list : A list of dictionaries. E.g. {'content': str, 'annotations': list(dict)}
    """
    re.compile(pattern)
    dataset = []

    book = open(filename)

    for line in book:
        line = line.strip()
        if line:
            line_data = dict()
            line_data["content"] = line
            line_data["annotations"] = get_annotations(line, pattern)
            dataset.append(line_data)

    return dataset
