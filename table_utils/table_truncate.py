from abc import ABC
import random
from typing import Dict
from transformers import PreTrainedTokenizer
from .table_linearize import TableLinearize

random.seed(42)


class TableTruncate(ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer, max_input_length: int):
        """
        The class `TableTruncate` is used to compress a table to fit in memory
        :param tokenizer: a huggingface transformer's tokenizer, to be used on BPE encoding to estimate expected tokens
        :param max_input_length: the maximum length of `question` and `table`, i.e., the max position id of a model
        """
        self.tokenizer = tokenizer
        self.max_length = max_input_length

    def truncate_table(self, table_content: Dict):
        """
        Given a table, return a truncated table with the same format.
        :return: no return value, but may modify table_content
        """
        pass


class CellLimitTruncate(TableTruncate):
    """
    Limit the maximum length of cell values in a table to truncate the overall length
    """

    def __init__(self, max_cell_length: int, **kwargs):
        super().__init__(**kwargs)
        self.max_cell_length = max_cell_length

    def truncate_table(self, table_content: Dict):
        for i, cell in enumerate(table_content["header"]):
            truncate_cell = self.truncate_cell(cell)
            if truncate_cell is not None:
                table_content["header"][i] = truncate_cell
        for row in table_content["rows"]:
            for i, cell in enumerate(row):
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    row[i] = truncate_cell

    def truncate_cell(self, cell_value):
        if cell_value.strip() != "":
            try_tokens = self.tokenizer.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
                retain_tokens = try_tokens[:self.max_cell_length]
                return self.tokenizer.convert_tokens_to_string(retain_tokens)
            else:
                return self.tokenizer.convert_tokens_to_string(try_tokens)
        return None


class RowRandomTruncate(TableTruncate):
    """
    The row truncate principle is straightforward: randomly select rows until the limit is reached.
    """

    def __init__(self, table_linearize: TableLinearize, include_title: bool, **kwargs):
        super().__init__(**kwargs)
        self.table_linearize = table_linearize
        self.include_title = include_title

    def truncate_table(self, table_content: Dict):
        remain_token_len = self.get_remain_token_len(table_content)

        # add rows randomly until the limit is reached
        rows = []
        while remain_token_len > 0 and len(table_content['rows']) > 0:
            row_idx = random.choice(range(len(table_content['rows'])))
            row = table_content['rows'].pop(row_idx)
            rows.append(row)

            row_string = self.table_linearize.process_row(row, 100)
            remain_token_len -= len(self.tokenizer.tokenize(row_string))

        table_content['rows'] = rows

    def get_remain_token_len(self, table_content: Dict):
        remain_token_len = self.max_length

        # if needed, add title and calculate occupied length
        if self.include_title:
            remain_token_len -= len(self.tokenizer.tokenize(table_content['title']))

        # add header and calculate occupied length
        # this is necessary, no need to consider remain_token_len
        header_string = self.table_linearize.process_header(table_content["header"])
        remain_token_len -= len(self.tokenizer.tokenize(header_string))

        return remain_token_len


class RowChooseTruncate(RowRandomTruncate):
    """
    The row truncate principle : first select the rows containing the answers,
    and then select the rows according to the degree of overlap between the rows tokens and the qa tokens
    until the limit is reached.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def truncate_table(self, table_content: Dict):
        if table_content['qa'] is None:
            # ff the table does not have related Q&A, add rows randomly until the limit is reached
            RowRandomTruncate.truncate_table(self, table_content)
        else:
            # otherwise, select rows according to relevance to Q&A
            # exclude title and header to calculate remain_token_len
            remain_token_len = self.get_remain_token_len(table_content)

            # calculate the tokens of answers and questions
            answers = []
            qa_tokens = set([])
            for question, answer in table_content['qa']:
                for ans in answer:
                    ans_tokens = self.tokenizer.tokenize(ans)
                    answers.append(' '.join(ans_tokens))
                    qa_tokens.update(ans_tokens)
                qa_tokens.update(self.tokenizer.tokenize(question))

            rows = []
            intersection_lens = []
            for row_idx, row_example in enumerate(table_content["rows"]):
                row_string = self.table_linearize.process_row(row_example, 100)
                row_tokens = self.tokenizer.tokenize(row_string)
                token_length = len(row_tokens)
                # determine whether the current row contains an answer
                in_flag = False
                for answer in answers:
                    if answer in ' '.join(row_tokens):
                        in_flag = True
                        break
                # if the current row contains an answer, add it directly to the table_string
                if in_flag:
                    rows.append(table_content['rows'][row_idx])
                    remain_token_len -= token_length
                    # if the limit has been reached, return the table_string directly
                    if remain_token_len <= 0:
                        table_content['rows'] = rows
                        return
                # otherwise, calculate the length of the current row overlapping with the qa_tokens
                else:
                    row_tokens = set(row_tokens)
                    intersection_len = len(row_tokens.intersection(qa_tokens))
                    intersection_lens.append((intersection_len, row_idx, token_length))

            # add rows in order of intersection_len
            for _, row_idx, token_length in sorted(intersection_lens, key=lambda x: x[0], reverse=True):
                rows.append(table_content['rows'][row_idx])
                remain_token_len -= token_length
                # if the limit has been reached, return the table_string directly
                if remain_token_len <= 0:
                    break
            table_content['rows'] = rows
