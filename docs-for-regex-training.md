# Regexes

You will want to make a `re.compile()` for each of these or you could combine them with a pipe `|` and bring them all together into one. For these, I recommend adding the `ignorecase` parameter.

```
Iliad\s\d{1,2}\.\d{1,4}
Il\.*\s\d{1,2}\.\d{1,4}
Iliad\s.[ivxlcdm]*\.\s*\d{1,4}
Il\.*\s.[ivxlcdm]*\.\s*\d{1,4}
book\s*.[ivxlcdm]\.\sline\s*\d{1,4}  # Note: this one doesn't say Iliad! Could be a good idea or terrible.
```

Let's talk about whether to include the the `Il.` or `Iliad` at the beginning of these.


# Documents to scan

This page ("Collection" in Archive.org's terms) has lots of old scholarly journals: https://archive.org/details/jstor_ejc?query=iliad&sin= . It may be a good place to start looking.


## Examples I looked at to make the regexes

Remember that for this task, you only need the raw text, not .pdf or .epub versions.

- https://archive.org/stream/jstor-4388167/4388167_djvu.txt
- https://archive.org/stream/jstor-3286678/3286678_djvu.txt
- https://archive.org/stream/jstor-3287794/3287794_djvu.txt
- https://archive.org/stream/jstor-20495500/20495500_djvu.txt


## Difficult examples for later!

Sometimes in old texts scholars will use Greek letters for Book numbers:

- https://archive.org/stream/jstor-289485/289485_djvu.txt
- https://archive.org/details/jstor-3287873/page/n3/mode/2up
- https://archive.org/details/jstor-849678/page/n3/mode/2up
- 
