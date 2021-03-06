import ast
import json
import re
from typing import Any, Union, TextIO

from bs4 import BeautifulSoup
from bs4.element import Tag
from internetarchive import search_items, get_item, Search
import requests
from requests import Response


def do_search(
    keyword_string: str, fulltext: bool = True, print_res: bool = True
) -> Search:
    """Execute search, return Archive library's ``Search`` object."""
    search_res: Search = search_items(keyword_string, full_text_search=fulltext)
    if print_res:
        print("Search string:", keyword_string)
        print("Results:", search_res.num_found)
    return search_res


def get_txt_url(book_url: str) -> str:
    """Scrape the text file url from the web page
    directed by the input ``book_url``.
    """
    book_web_page: Response = requests.get(book_url)
    book_web_page_soup: BeautifulSoup = BeautifulSoup(book_web_page.text, "html.parser")
    txt_url: str = ""
    for link in book_web_page_soup.find_all("a", "stealth"):
        txt_url_candidate: str = link.get("href")
        if txt_url_candidate.endswith("_djvu.txt"):
            txt_url = txt_url_candidate
    return txt_url


def get_annotations(text, pattern) -> list[dict[str, Union[int, str]]]:
    """Helper function for prepare_data

    Args:
        text (str): Input string
        pattern (regex expression): pattern we are looking for from the input string

    Returns:
        list: A list of dictionaries recording matches we found.
        E.g. [{'start': int, 'end': int, 'label': str}, ]
    """
    annotations: list[dict[str, Union[int, str]]] = list()
    # find all strings matching the input pattern.
    for match in re.finditer(pattern, text):
        label_dic: dict[str, Union[int, str]] = dict()
        label_dic["start"] = match.start()
        label_dic["end"] = match.end()
        label_dic["label"] = "Citation"  # Entity starting with a capital letter.
        annotations.append(label_dic)
    return annotations


def prepare_data(
    search_res: Search, pattern: str, num_of_pos: int = 1000, num_of_neg: int = 1000
) -> None:
    """From all `search_res`, scrape `num_of_pos` positive lines that include matches to the `pattern`
    and `num_of_neg` negative lines without any matches found.
    """
    positive: int = 0
    negative: int = 0
    book_count: int = 0
    pos_instances = open(
        f"pos_neg_instances/pos_instances_{num_of_pos}.txt", "w"
    )
    neg_instances = open(
        f"pos_neg_instances/neg_instances_{num_of_neg}.txt", "w"
    )
    # Iterate over search object
    for item in search_res:
        identifier = item["fields"]["identifier"][0]
        book_url = f"https://archive.org/details/{identifier}"
        book_count += 1
        # Get the url to .txt file on the archive.org webpage
        txt_url = get_txt_url(book_url)
        if txt_url:
            # Access to the text
            txt_url: str = f"https://archive.org{txt_url}"
            txt_web_page: Response = requests.get(txt_url)
            # Parse book context from the link
            txt_web_page_soup: BeautifulSoup = BeautifulSoup(txt_web_page.text, "html.parser")
            book_content: Tag = txt_web_page_soup.find_all("pre")[0]
            # Go through lines of the book and find matches to the regex pattern
            lines: list[str] = book_content.text.split("\n")
            for line in lines:
                if line != "":
                    line_data: dict[str, Union[str, list[dict[str, Union[int, str]]]]] = dict()
                    line_data["content"] = line
                    line_data["annotations"] = get_annotations(line, pattern)
                    # positive instance found
                    if len(line_data["annotations"]) != 0 and positive < num_of_pos:
                        pos_instances.write(str(line_data) + "\n")
                        positive += 1
                    # negative instance found
                    elif len(line_data["annotations"]) == 0 and negative < num_of_neg:
                        neg_instances.write(str(line_data) + "\n")
                        negative += 1
        if positive == num_of_pos and negative == num_of_neg:
            break
    pos_instances.close()
    neg_instances.close()
    print(f"Successfully got {positive} positive data and {negative} negative data by scraping {book_count} books!")
    print(f"Positive instances are saved at: pos_neg_instances/pos_instances_{num_of_pos}.txt")
    print(f"Negative instances are saved at: pos_neg_instances/neg_instances_{num_of_neg}.txt")
    return None


def get_scraped_dataset_size(file_name: str) -> int:
    """
    Return the number of instances saved in the text file path.
    """
    file: TextIO = open(file_name, "r")
    line_count: int = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    return line_count


def load_scraped_data(file_name) -> list[dict[str, Union[str, dict[str, Union, str, int]]]]:
    """
    Load scraped pos/neg instances data from the input text file path.
    """
    labeled_data: list[dict[str, Union[str, dict[str, Union, str, int]]]] = list()
    # load instances
    with open(file_name) as instances_file:
        lines = [line.strip() for line in instances_file.readlines()]
    for line in lines:
        line = json.dumps(ast.literal_eval(line))
        labeled_data.append(json.loads(line))
    return labeled_data
