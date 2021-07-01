from dataclasses import dataclass

# Author, Work, Book, Line, Commentary Name, URL
# "Homer", "Iliad", 1, 45, "firstthreebooks03homegoog", "https://archive.org/details/firstthreebooks03homegoog/page/n210/mode/2up"

@dataclass
class SingleLineComment:
    book_number: int
    line_number: int
    archive_page_number: int
    archive_link: str
    line_commentary: str
    section_commentary: str

@dataclass
class Commentary:
    modern_author: str
    ancient_author: str
    ancient_work: str
    modern_title: str
    archive_id: str
    archive_url: str
    single_line_comment: list[SingleLineComment] = None

    def query_line_comm(self, book_num, line_num, sec_comm = False):
        try:
            for line in self.single_line_comment:
                if line.book_number == book_num and line.line_number == line_num:
                    if sec_comm:   
                        return line.section_commentary
                    else:
                        return line.line_commentary
        except ValueError:
            print("Sorry no commentary found.")