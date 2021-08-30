# Seymour, first 3 books

Commentary begins: https://archive.org/details/firstthreebooks03homegoog/page/n123/mode/2up

Things to notice:

- For book start: "FIRST BOOK OF THE ILIAD."
    - Hopefully, there is a "SECOND BOOK OF THE ILIAD."
- Comment Type 1: A group of lines, eg 1-7 (a range of lines)
- Comment Type 2: A single line
- Observation: Single line comments appear in order
- Observation: Comments for Type 1 seem to cover every line; eg, 1-7, then 8-52, 53-100 (etc)
- Observation: Type 2 comments appear within the range of the Type 1 comment

Homer, Il., book 1, line 1: "firstthreebooks03homegoog/page/n123"


# Open questions

- What format to download and search over locally? (See "DOWNLOAD OPTIONS" on the archive page)

# Example algorithm

- Divide commentary into 4 parts:
  - Intro (ignore)
  - Bk 1
  - Bk 2
  - Bk 3
- W/in the three books, find the Type 1 range comments
  - Eg, 1-7, 8-52
- W/in each range comment, find each Type 2 comment
- regex example for lines: \n + \n + int of length 1, 2, or 3 chars + "." + " " + begin various characters
- the range comments will be similar to above, but with a "-" inside the integer


# Language models

- https://huggingface.co/transformers/model_doc/bert.html
