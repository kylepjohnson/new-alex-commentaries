"""Do Boolean search on Archive.org's API."""

from internetarchive import search_items, get_item, Search


def do_search(
    keyword_string: str, fulltext: bool = True, print_res: bool = True
) -> Search:
    """Execute search, return Archive library's
    ``Search`` object.
    """
    search_obj: Search = search_items(keyword_string, full_text_search=fulltext)
    if print_res:
        print("Search string:", keyword_string)
        print("Results:", search_obj.num_found)
    return search_obj


def try_search():
    search_obj: Search = search_items("homer commentary")
    # for item in search_obj:
    #     print(item['identifier'])
    print(search_obj.num_found)

    # full text search
    search_fts: Search = search_items("homer commentary", full_text_search=True)
    # for item in search_fts:
    #     print(item['identifier'])
    print(search_fts.num_found)

    # search by added date
    search_date: Search = search_items(
        "(homer commentary) AND date:[2019-01-01 TO 2020-01-01]"
    )
    for item in search_date:
        print(item["identifier"])
    print(search_date.num_found)

    # Search by media type
    search_field: Search = search_items(
        "(homer commentary) AND mediatype:(software)", fields="avg_rating"
    )
    for item in search_field:
        print(item)
    print(search_field.num_found)


def try_download():
    """
    Download All the META XML Files from a Collection
    """
    search = search_items(
        "(homer commentary) AND mediatype:(texts) AND date:[2019-01-01 TO 2020-01-01]"
    )
    num = 0
    for result in search:
        num = num + 1
        itemid = result["identifier"]
        print("Downloading " + str(num) + "\t" + itemid + " ...")
        item = get_item(itemid)
        meta = item.get_file(itemid + "_meta.xml")
        meta.download()


if __name__ == "__main__":
    # Run examples with these two:
    # try_search()
    # try_download()

    # these two add up to 598,241
    query_il = "homer iliad"  # 312,6018
    query_od = "homer odyssey"  # 285,633

    # With the AND results are identical
    query_il2 = "homer AND iliad"  # 312,608
    query_od2 = "homer AND odyssey"  # 285,633

    # Why then does this add to only 430,055?
    query_il_od2 = "homer AND (iliad OR odyssey)"  # 430,055

    # why? something must be wrong
    query_il_od3 = "homer AND iliad OR odyssey"  # 1,512,088

    query_il_od = "homer iliad OR odyssey"  # 1,512,073
    query_with_date = "(homer iliad) AND date:[2019-01-01 TO 2020-01-01]"  # 629
    il_od = "iliad OR odyssey"  # 771,646
    do_search(keyword_string=il_od)
    # do_search(keyword_string=query_od2)
