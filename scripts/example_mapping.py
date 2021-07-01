import re



def generate_mapping(fileName):
    """[This function returns a mapping of Homer lines to it's corresponding commentaries]

    Args:
        fileName ([str]): [Name of text file of the commentary book]

    Returns:
        [dict]: [Mapping with the following data structure]
        mapping = {Book1: {Line1: {Page: int, Sec_comm: str, Line_comm: str} }, Book2:, Book3:, ...}
    """
    mapping = {}

    with open(fileName) as f:
        lines = f.read()

    paragraphs = lines.split("\n\n")

    comm_start = 0
    curr_book = ''
    curr_line = ''
    curr_page = 0
    sec_comm = ''
    # sec_min = 0
    # sec_max = 0

    # Commentaries begin
    for i in range(len(paragraphs)):
        if re.search('^COMMENTARY. $', paragraphs[i]):
            comm_start = i
            break
    comm = paragraphs[comm_start:]

    for i in range(len(comm)):
        if comm[i] == '':
            continue
        
        # Commentary begins
        elif comm[i] == 'COMMENTARY. ' and curr_page == 0:
            curr_page = 1
        elif comm[i] == 'COMMENTARY. ' and curr_page != 0:
            continue

        # Change page
        elif (re.search('^\d+ $', comm[i]) or re.search('^\s\d+\s$', comm[i])) and comm[i-1] == '' and comm[i+1] == '':
            curr_page += 1

        # Book1
        elif comm[i].startswith('FIRST BOOK') and curr_book == '':
            curr_book = 'Book1'
            mapping[curr_book] = {}
        elif comm[i].startswith('FIRST BOOK') and curr_book != '':
            continue  
        
        #  Edge Case 1 (28-32 = 11-15)
        elif curr_book and sec_comm and re.search('^\d+-\d+ = \d+-\d+', comm[i]):
            continue

        # Secotion Commentaries
        elif curr_book and re.search('^\d+-\d+. ', comm[i]):
            sec_comm = comm[i]
            # sec_range = re.search('^\d+-\d+. ', comm[i]).group()[:-2].split('-')
            # sec_min = int(sec_range[0])
            # sec_max = int(sec_range[1])

        # Line Comm Edge Case 2 (324 = 137, 451 f. = 37 f . )
        elif curr_book and sec_comm and (re.search('^\d+ = \d+', comm[i]) or (re.search('^\d+ f. =', comm[i]))):
            curr_line = 'Line' + re.search('^\d+. ', comm[i]).group()[:-1]
            mapping[curr_book][curr_line] = {}
            mapping[curr_book][curr_line]['Page'] = curr_page
            mapping[curr_book][curr_line]['Sec_comm'] = sec_comm
            mapping[curr_book][curr_line]['Line_comm'] = comm[i]

        # Line Commentaries
        elif curr_book and sec_comm and re.search('^\d{1,3}.\s', comm[i]):
            curr_line = 'Line' + re.search('^\d+. ', comm[i]).group()[:-2]
            mapping[curr_book][curr_line] = {}
            mapping[curr_book][curr_line]['Page'] = curr_page
            mapping[curr_book][curr_line]['Sec_comm'] = sec_comm
            mapping[curr_book][curr_line]['Line_comm'] = comm[i]

        # Book2
        elif comm[i].startswith('SECOND BOOK') and curr_book == 'Book1':
            curr_book = 'Book2'
            mapping[curr_book] = {}
        elif comm[i].startswith('SECOND BOOK') and curr_book != 'Book1':
            continue

        # Book3
        elif comm[i].startswith('THIRD BOOK') and curr_book == 'Book2':
            curr_book = 'Book3'
            mapping[curr_book] = {}
        elif comm[i].startswith('SECOND BOOK') and curr_book != 'Book2':
            continue

        elif not mapping[curr_book].get(curr_line):
            continue
        else:
            mapping[curr_book][curr_line]['Line_comm'] += ('\n'+ comm[i])

    return mapping

def string_mapping(fileName, book, line, sec_comment=False):

    """[This function returns the commentaries of the queried book/line from a given text file.]

    Args:
        fileName ([str]): [Name of the text file]
        book ([int]): [Book number]
        line ([int]): [Line number]
        sec_comment (bool, optional): [Returns the section commentary that the quried line is in]. Defaults to False.

    Returns:
        [str]: [Commentaries]
    """
    mapping = generate_mapping(fileName)
    search_book = 'Book'+ str(book)
    search_line = 'Line' + str(line)
    try:
        if sec_comment:
            return mapping[search_book][search_line]['Sec_comm']
        else:
            return mapping[search_book][search_line]['Line_comm']
    except KeyError:
        print("Sorry no commentary found.")
        

def page_mapping(fileName, book, line):

    """[This function returns the web link to the commentaries of the queired Homer book/line]

    Args:
        fileName ([str]): [Name of the text file]
        book ([int]): [Book number]
        line ([int]): [Line number]

    Returns:
        [str]: [Web page link]
    """
    mapping = generate_mapping(fileName)
    search_book = 'Book'+ str(book)
    search_line = 'Line' + str(line)
    try:
        page_num = str(mapping[search_book][search_line]['Page'] + 122)
        link = 'https://archive.org/details/firstthreebooks03homegoog/page/n'+page_num+'/mode/2up'
        return link
    except ValueError:
        print("Sorry no commentary found.")
        

if __name__ == "__main__":
    fileName = "../example-texts/seymour-il-1-3.txt"
    #print(generate_mapping(fileName))
    print(string_mapping(fileName, 2, 45))
    #print(string_mapping(fileName, 2, 45, True))
    print(page_mapping(fileName, 2, 45))
    