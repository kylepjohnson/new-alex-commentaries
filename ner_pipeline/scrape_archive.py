import requests
import re
from bs4 import BeautifulSoup
from internetarchive import search_items, get_item, Search


def do_search(keyword_string: str, fulltext: bool = True, print_res: bool = True) -> Search:
    """Execute search, return Archive library's
    ``Search`` object.
    """
    search_res: Search = search_items(keyword_string, full_text_search=fulltext)
    if print_res:
        print("Search string:", keyword_string)
        print("Results:", search_res.num_found)
    return search_res


def get_txt_url(book_url: str) -> str:
    """Scrape the text file url from the web page 
    directed by the input ``book_url``.
    """
    book_web_page = requests.get(book_url)
    book_web_page_soup = BeautifulSoup(book_web_page.text, 'html.parser')
    txt_url = ""
    for link in book_web_page_soup.find_all('a', 'stealth'):
        txt_url_candidate = link.get('href')
        if txt_url_candidate.endswith('_djvu.txt'):
            txt_url = txt_url_candidate   
    return txt_url


# def generate_raw_dataset(search_res: Search, num_of_searches: int = 100) -> list:
#     """Scrape the text from ``num_of_searches`` searching results.
#     """
#     raw_data = []
#     i = 0
#     count = 0
#     for item in search_res:
#         if i < num_of_searches:
#             book_url = "https://archive.org/details/" + item["fields"]["identifier"][0]
#             # Get the url to .txt file
#             txt_url = get_txt_url(book_url)
#             if txt_url:
#                 txt_url = 'https://archive.org' + txt_url
#                 # Access to the text
#                 txt_web_page = requests.get(txt_url)
#                 txt_web_page_soup = BeautifulSoup(txt_web_page.text, 'html.parser')
#                 book_content = txt_web_page_soup.find_all('pre')[0]
#                 lines = book_content.text.split('\n')
#                 lines = [line for line in lines if line != '']
#                 raw_data += lines
#                 count += 1
#         else:
#             break
#         i += 1
#     print("Successfully scraped " + str(count) + " books out of "+ str(num_of_searches) + " searching results!")
#     return raw_data

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
        label_dic["label"] = "Citation"  # Entity starting with a capital letter.
        annotations.append(label_dic)
    return annotations


def prepare_data(search_res: Search, pattern: str, num_of_pos: int = 1000, num_of_neg: int = 1000):
    re.compile(pattern)
    dataset = []
    positive = 0
    negative = 0
    for item in search_res:
        book_url = "https://archive.org/details/" + item["fields"]["identifier"][0]
        # Get the url to .txt file
        txt_url = get_txt_url(book_url)
        if txt_url:
            txt_url = 'https://archive.org' + txt_url
            # Access to the text
            txt_web_page = requests.get(txt_url)
            txt_web_page_soup = BeautifulSoup(txt_web_page.text, 'html.parser')
            book_content = txt_web_page_soup.find_all('pre')[0]
            lines = book_content.text.split('\n')
            for line in lines:
                if line != '':
                    line_data = dict()
                    line_data["content"] = line
                    line_data["annotations"] = get_annotations(line, pattern)
                    if len(line_data["annotations"]) != 0 and positive < num_of_pos:
                        dataset.append(line_data)
                        positive += 1
                    elif len(line_data["annotations"]) == 0 and negative < num_of_neg:
                        dataset.append(line_data)
                        negative += 1
        if positive == num_of_pos and negative == num_of_neg:
            break
    print("Successfully got " + str(positive) + " positive data and "+ str(negative) + " negative data!")
    return dataset
                        






