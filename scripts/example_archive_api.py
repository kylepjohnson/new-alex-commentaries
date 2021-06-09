
from internetarchive import search_items
from internetarchive.search import Search


def try_search():
    search_obj: Search = search_items("homer commentary")
    for item in search_obj:
        print(item['identifier'])


if __name__ == "__main__":
    try_search()
