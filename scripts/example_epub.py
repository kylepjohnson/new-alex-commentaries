import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from data_structure import SingleLineComment
from data_structure import Commentary
import re

def get_paragraphs(book):
    """[Find all html tags containing commentaries and web page links from a epub file]

    Args:
        epub_path ([ebooklib.epub.EpubBook]): [An EpubBook object]

    Returns:
        [list]: [A list of strings recording commentaries and web page links]
    """
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())
    soup = BeautifulSoup(chapters[1], 'html.parser')
    paragraphs = [para for para in soup.find_all(['p','span'])]
    return paragraphs

def mapping(comm, paragraphs):
    """[A helper function for creat_mapping. It maps each line in an ancient Greek literature to its corresponding commentary.]

    Args:
        comm ([Commentary Object]): [A Commentary object defined in data_structure.py]
        paragraphs ([list]): [A list of strings recording commentaries and web page links]

    Returns:
        [None]: [This function changes the single_line_comment variable of the input Commentary Object]
    """
    book_dict = {'FIRST': 1, 'SECOND': 2, 'THIRD': 3}
    comm.single_line_comment = []
    archive_url = comm.archive_url

    curr_book = 0
    curr_page = 0
    curr_sec_comm = ''
    start = False

    for para in paragraphs:

        if para.name == 'p':
            text = para.get_text()
            if text == 'COMMENTARY. ' and not start:
                start = True
                
            # Commentary Begins
            if start:
                
                # Book Number
                if re.search('BOOK\sOF\sTHE\sILIAD.\s$', text):
                    curr_book = book_dict[text.split()[0]]
                    
                # Section Comm (Eg: 1-7. ) and Edge Case (Eg: 28-32 = 11-15)
                elif re.search('^\d{1,3}-\d{1,3}\s=\s\d{1,3}-\d{1,3}', text) or re.search('^\d{1,3}-\d{1,3}.\s', text):
                    curr_sec_comm = text
                
                # Line Comm Edge Case (Eg: 324 = 137, 451 f. = 37 f . )
                elif (re.search('^\d{1,3}\s=\s\d{1,3}', text) or (re.search('^\d{1,3}\sf.\s=', text))):
                    line_num = int(re.search('^\d+. ', text).group()[:-1])
                    comm.single_line_comment.append(SingleLineComment(book_number = curr_book, 
                                                                            line_number = line_num,
                                                                            archive_page_number = curr_page,
                                                                            archive_link = curr_link,
                                                                            line_commentary = text,
                                                                            section_commentary = curr_sec_comm))
                # Line Comm (Eg: 23. )
                elif re.search('^\d{1,3}.\s', text):
                    line_num = int(re.search('^\d{1,3}.\s', text).group()[:-2])
                    comm.single_line_comment.append(SingleLineComment(book_number = curr_book, 
                                                                            line_number = line_num,
                                                                            archive_page_number = curr_page,
                                                                            archive_link = curr_link,
                                                                            line_commentary = text,
                                                                            section_commentary = curr_sec_comm))
        # Web Page Link
        elif para.name == 'span' and para.attrs['epub:type'] == "pagebreak":
            curr_page = int(para.attrs['id'])+1
            curr_link = archive_url + '/page/n'+str(curr_page)+'/mode/2up'
    return 

def creat_mapping(epub_path):
    """[Create the commentary mapping for the input epub book]

    Args:
        epub_path ([str]): [Path to the epub book file]

    Returns:
        [Commentary Object]: [A Commentary object defined in data_structure.py]
    """
    book = epub.read_epub(epub_path)
    comm = Commentary(modern_author=book.get_metadata('DC', 'creator')[1][0],
                         ancient_author=book.get_metadata('DC', 'creator')[0][0],
                         ancient_work="Iliad",
                         modern_title=book.get_metadata('DC', 'title')[0][0],
                         archive_id=book.get_metadata('DC', 'identifier')[0][0],
                         archive_url=book.get_metadata('DC', 'identifier')[1][0].split(': ')[1])

    paragraphs = get_paragraphs(book)
    mapping(comm, paragraphs)
    return comm


def query(epub_path, book_num, line_num, sec_comm=False):
    """[User can query the commentary of a specified line of the input epub book]

    Args:
        epub_path ([str]): [Path to the epub book file]
        book_num ([int]): [Ancient book number]
        line_num ([int]): [Ancient line number]
        sec_comm (bool, optional): [Section comment will be returned if sec_comm is True]. Defaults to False.

    Returns:
        [str]: [The corresponding commentary of the queried line]
    """
    mapping = creat_mapping(epub_path)
    comm = mapping.query_line_comm(book_num, line_num, sec_comm)
    return comm


if __name__ == "__main__":
    fileName = "../example-texts/firstthreebooks03homegoog_firstthreebooks03homegoog.epub"
    book_num = 1
    line_num = 2
    print(query(fileName, book_num, line_num))
    print(query(fileName, book_num, line_num, sec_comm=True))
    