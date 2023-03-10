import os
import re
import json
import random
from typing import Dict, List
from collections import defaultdict


def flatten_sql(headers: List, _ex_sql_struct: List):
    # [ "Keyword", "select", [] ], [ "Column", "c4", [] ]
    _encode_sql = []
    for _ex_tuple in _ex_sql_struct:
        keyword = str(_ex_tuple[1])
        assert '_' not in keyword
        # extra column, which we do not need in result
        if keyword == "w" or keyword == "from":
            continue
        elif re.fullmatch(r"c\d+(_.+)?", keyword):
            # only take the first part
            index_key = int(keyword.split("_")[0][1:]) - 1
            _encode_sql.append(headers[index_key])
        else:
            _encode_sql.append(keyword)
        # c4_list, replace it with the original one
        # if "_address" in keyword or "_list" in keyword:
        #     keyword = re.findall(r"c\d+", keyword)[0]

    return " ".join(_encode_sql)


def generate_sql_on_target_table(_sql_struct: List,
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

    _tgt_encode_sql, _tgt_answer, _tgt_exec_sql = "", [], ""
    src_map_to_tgt = {}
    for i in range(len(_sql_struct)):
        keyword_type, keyword_name, _ = _sql_struct[i]
        # if there has established the mapping, directly replace it
        if keyword_name in src_map_to_tgt:
            pass
        # otherwise, if it is a column
        elif keyword_type == "Column":
            src_col_name = keyword_name.split("_")[0]
            valid_tgt_col_cands = valid_col_cands[src_col_name]
            # such a template cannot apply on the target database
            if len(valid_tgt_col_cands) == 0:
                return None
            # any column has at least one valid target
            src_map_to_tgt[keyword_name] = random.choice(valid_tgt_col_cands)
        # if it is a value (number)
        elif keyword_type in ["Literal.String", "Literal.Number"]:
            src_val_name = str(keyword_name).strip("'")
            if src_val_name in src_val_records:
                src_val_pos = src_val_records[src_val_name]
                # existing src names with the table position
                src_used_col = set([_.split("_")[0] for _ in src_map_to_tgt.keys()])
                src_val_col = list(src_val_pos & src_used_col)
                # if src_val_col is empty, skip
                if len(src_val_col) != 0:
                    src_val_col = random.choice(src_val_col)
                    # take the mapping column
                    if src_val_col not in src_map_to_tgt:
                        for src_col_name in src_map_to_tgt.keys():
                            if src_val_col in src_col_name:
                                src_val_col = src_col_name
                    tgt_val_col = src_map_to_tgt[src_val_col]
                    tgt_val_col_ind = int(tgt_val_col.split("_")[0][1:]) - 1
                    # find the content, randomly take one value as the replacement
                    tgt_rand_val = tgt_col_content[tgt_val_col_ind]
                    tgt_rand_val = random.choice(tgt_rand_val)
                    try:
                        src_map_to_tgt[keyword_name] = int(tgt_rand_val)
                    except ValueError:
                        src_map_to_tgt[keyword_name] = "'{}'".format(tgt_rand_val)
            else:
                if keyword_type == "Literal.Number":
                    random_val = str(random.randint(0, 2023))
                else:
                    random_val = "'{}'".format(random.choice(tgt_val_list))
                src_map_to_tgt[keyword_name] = random_val

            if keyword_name in src_map_to_tgt and \
                    keyword_type in ["Column", "Literal.String", "Literal.Number"]:
                _sql_struct[i][1] = src_map_to_tgt[keyword_name]

    _tgt_encode_sql = flatten_sql(_tgt_table['header'], _sql_struct)
    return _tgt_encode_sql


class Generator:

    def __init__(self):
        self.examples = json.load(open('squall/data.json'))
        self.src_tables = {}
        dir_path = 'squall/tables'
        for table_path in os.listdir(dir_path):
            table = json.load(open(os.path.join(dir_path, table_path)))
            self.src_tables[table_path[:-5]] = table

    def generate(self, tgt_tables: List[Dict], ques_per_passage: int = 1):
        qa_cnt = 0
        qas = []
        for tgt_table in tgt_tables:
            sql_cnt = 0
            while sql_cnt < ques_per_passage:
                example = random.choice(self.examples)
                sql_struct = example["sql"]
                src_table = self.src_tables[example['tbl']]
                sql = generate_sql_on_target_table(_sql_struct=sql_struct,
                                                   _src_table=src_table,
                                                   _tgt_table=tgt_table)
                if sql is not None:
                    sql_cnt += 1
                    qa_cnt += 1
                    qas.append({
                        'id': "genS" + str(qa_cnt),
                        'question': sql,
                        'answer': [],
                        'table_id': [tgt_table['id']]
                    })

        return qas
