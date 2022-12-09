# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatted sequence
"""
import abc
from typing import Dict, List


class TableLinearize(abc.ABC):
    """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """
    def __init__(self, include_title: bool):
        self.include_title = include_title

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatted sequence.
        """
        pass

    def process_header(self, headers: List) -> str:
        """
        Given a list of headers, TableLinearize aims at converting it into a flatted sequence.
        """
        pass

    def process_row(self, row: List, row_index: int = 0) -> str:
        """
        Given a row, TableLinearize aims at converting it into a flatted sequence.
        """
        pass


class BasicTableLinearize(TableLinearize):
    """
    FORMAT: (title) col1 col2 col val11 val12 val13 val21  val22 val23 ...
    """
    def __init__(self, include_title: bool):
        super().__init__(include_title)

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatted sequence without special symbols.
        """
        if self.include_title:
            _table_str = table_content['title'] + " "
        else:
            _table_str = ""
        # process header
        _table_str += self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example) + " "
        return _table_str.strip()

    def process_header(self, headers: List) -> str:
        """
        Given a list of headers, TableLinearize aims at converting it into a flatted sequence with special symbols.
        """
        return " ".join(headers)

    def process_row(self, row: List, row_index: int = 0) -> str:
        """
        Given a row, TableLinearize aims at converting it into a flatted sequence without special symbols.
        """
        return " ".join(row)


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: (title) col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """
    def __init__(self, include_title: bool):
        super().__init__(include_title)

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatted sequence with special symbols.
        """
        if self.include_title:
            _table_str = table_content['title'] + " "
        else:
            _table_str = ""
        # process header
        _table_str += self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"], start=1):
            _table_str += self.process_row(row_example, row_index=i) + " "
        return _table_str.strip()

    def process_header(self, headers: List) -> str:
        """
        Given a list of headers, TableLinearize aims at converting it into a flatted sequence with special symbols.
        """
        return " col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int = 0) -> str:
        """
        Given a row, TableLinearize aims at converting it into a flatted sequence with special symbols.
        """
        return " row " + str(row_index) + " : " + " | ".join(row)
