from typing import Dict, List
from .table_linearize import TableLinearize, BasicTableLinearize, IndexedRowTableLinearize
from .table_truncate import TableTruncate, CellLimitTruncate, RowRandomTruncate, RowChooseTruncate
from transformers import PreTrainedTokenizer


class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate]):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs

    def process_table(self, table_content: Dict) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content)
        # linearize a table into a string
        return self.table_linearize_func.process_table(table_content)


def get_processor(max_cell_length: int, max_input_length: int, tokenizer: PreTrainedTokenizer,
                  include_title: bool, index_row: bool, row_choose: bool):
    if index_row:
        table_linearize_func = IndexedRowTableLinearize(include_title)
    else:
        table_linearize_func = BasicTableLinearize(include_title)
    table_truncate_funcs = [CellLimitTruncate(tokenizer=tokenizer,
                                              max_input_length=max_input_length,
                                              max_cell_length=max_cell_length)]
    if row_choose:
        table_truncate_funcs.append(RowChooseTruncate(tokenizer=tokenizer,
                                                      max_input_length=max_input_length,
                                                      table_linearize=table_linearize_func,
                                                      include_title=include_title))
    else:
        table_truncate_funcs.append(RowRandomTruncate(tokenizer=tokenizer,
                                                      max_input_length=max_input_length,
                                                      table_linearize=table_linearize_func,
                                                      include_title=include_title))
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor
