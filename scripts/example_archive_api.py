
from internetarchive import search_items
from internetarchive.search import Search
import internetarchive
import json

def try_search():
    
    search_obj : Search = search_items("homer commentary")
    # for item in search_obj:
    #     print(item['identifier'])
    print(search_obj.num_found)

    # full text search
    search_fts : Search = search_items("homer commentary", full_text_search = True)
    # for item in search_fts:
    #     print(item['identifier'])
    print(search_fts.num_found)

    # search by added date
    search_date : Search = search_items("(homer commentary) AND date:[2019-01-01 TO 2020-01-01]")
    for item in search_date:
        print(item['identifier'])
    print(search_date.num_found)

    # Search by media type
    search_field : Search = search_items("(homer commentary) AND mediatype:(software)", fields="avg_rating")
    for item in search_field:
        print(item)
    print(search_field.num_found)


def try_download():
    """
    Download All the META XML Files from a Collection
    """
    search = internetarchive.search_items("(homer commentary) AND mediatype:(texts) AND date:[2019-01-01 TO 2020-01-01]")
    num = 0
    for result in search:
        num = num + 1
        itemid = result['identifier']
        print("Downloading " + str(num) + '\t' + itemid + " ...")
        item = internetarchive.get_item(itemid)
        meta = item.get_file(itemid+'_meta.xml')
        meta.download()


if __name__ == "__main__":
    try_search()
    #try_download()
