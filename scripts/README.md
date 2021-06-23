# Searching

## Function

Search for items on Archive.org using `search_items()` function.
  - Inputs: **query** (str), **fields** (list), **full_text_search** (bool), etc.
  - Return: a interable `Search` object (a collection of search results).
  - Example:  
    ```
    >>> from internetarchive import search_items
    >>> search = search_items('homer commentary')
    >>> print(search.num_found)
    107
    ```
    
Get the `identfier' of each search result:

```
>>> for item in search:
...     print(item['identifier'])

```
Access search results' attributes by turning them to `Item` objects: 

```
>>> for item in searchiter_as_items():
...     print(item.metadata)
```

## Advanced Search

The `query` input can be customized by specifying variables using simple syntax.

For example, we can search for items related to "homer commentary" within the data range from 2019-01-01 to 2020-01-01.
```
>>> from internetarchive import search_items
>>> search = search_items("(homer commentary) AND date:[2019-01-01 TO 2020-01-01]")
>>> print(search.num_found)
4
```
Search customization details are available here: [link](https://archive.org/advancedsearch.php).

A Few Advanced Search Tips: [link](https://blog.archive.org/2017/04/16/a-few-advanced-search-tips/)


## Metadata Read API

Retrieves metadata for items on archive.org and returns metadata in `JSON`.

```
>>> curl http://archive.org/metadata/:identifier
```

Access specific metadata elements:
```
http://archive.org/metadata/:identifier/metadata
http://archive.org/metadata/:identifier/server
http://archive.org/metadata/:identifier/files_count
http://archive.org/metadata/:identifier/files?start=1&count=2
http://archive.org/metadata/:identifier/metadata/collection
http://archive.org/metadata/:identifier/metadata/collection/0
http://archive.org/metadata/:identifier/metadata/title
http://archive.org/metadata/:identifier/files/0/name
```

The Internet Archive Metadata API: [link](http://blog.archive.org/2013/07/04/metadata-api/)

Metadata Attributes: [Link](https://archive.org/services/docs/api/metadata-schema/index.html#)

# Mapping
`example_mapping.py` is a program to generate commentary mapping of a sample text book saved in `../example-texts/seymour-il-1-3.txt`, which is downloaded from Archive.org: [Link](https://archive.org/details/firstthreebooks03homegoog/page/n123/mode/2up)
