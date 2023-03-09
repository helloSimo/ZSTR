import os
import json
import random
from typing import Dict, List
from collections import defaultdict


class Generator:

    def __init__(self):
        self.examples = json.load(open('squall/data.json'))
        self.src_tables = {}
        dir_path = 'squall/tables'
        for table_path in os.listdir(dir_path):
            table = json.load(open(os.path.join(dir_path, table_path)))
            self.src_tables[table_path[:-5]] = table

    def apply_sql_on_target_table(self, _sql_struct: List,
                                  _src_table: Dict, _tgt_table: Dict):
        # find types of columns in _origin_table
        src_col_content = list(map(list, zip(*_src_table["rows"])))
        src_val_records = defaultdict(lambda: set())
        for i in range(len(src_col_content)):
            col_example = src_col_content[i]
            for j in range(len(col_example)):
                # map from value to a column
                src_val_records[str(col_example[j])].add("c" + str(i + 1))

        # use for sample value to replace the original one
        tgt_col_content = list(map(list, zip(*_tgt_table["rows"])))
        tgt_val_list = []
        for example in tgt_col_content:
            tgt_val_list.extend([_ for _ in example if _ != "none"])

        tgt_col_alias = _tgt_table["alias"]
        tgt_num_cols, tgt_str_cols = [], []
        for i in range(len(_tgt_table["header"])):
            if _tgt_table["types"][i] == 'number':
                tgt_num_cols.append("c" + str(i + 1))
            else:
                tgt_str_cols.append("c" + str(i + 1))

        # find the column types
        valid_col_cands = {}
        src_col_num = {}
        for i, src_type in enumerate(_src_table["types"]):
            col_name = "c" + str(i + 1)
            if src_type == 'number':
                valid_col_cands[col_name] = tgt_num_cols
                src_col_num[col_name] = True
            else:
                valid_col_cands[col_name] = tgt_str_cols
                src_col_num[col_name] = False


    def generate(self, tgt_table: Dict[str: str], ques_per_passage: int = 1):
        cnt = 0
        while cnt < ques_per_passage:
            example = random.choice(self.examples)
            sql_struct = example["sql"]
