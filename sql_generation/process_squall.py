import os
import json
from tqdm import tqdm
from typing import Dict


def process_table_structure(_wtq_table_content: Dict):
    # remove id and agg column
    headers = [_.replace("\n", " ").lower() for _ in _wtq_table_content["headers"][2:]]
    header_map = {}
    for i in range(len(headers)):
        header_map["c" + str(i + 1)] = headers[i]
    header_types = _wtq_table_content["types"][2:]

    all_headers = []
    all_header_types = []
    vertical_content = []
    for column_content in _wtq_table_content["contents"][2:]:
        # only take the first one
        for i in range(len(column_content)):
            column_alias = column_content[i]["col"]
            # do not add the numbered column
            if "_number" in column_alias:
                continue
            vertical_content.append([str(_).replace("\n", " ").lower() for _ in column_content[i]["data"]])
            if "_" in column_alias:
                first_slash_pos = column_alias.find("_")
                column_name = header_map[column_alias[:first_slash_pos]] + " " + \
                              column_alias[first_slash_pos + 1:].replace("_", " ")
            else:
                column_name = header_map[column_alias]
            all_headers.append(column_name)
            if column_content[i]["type"] == "TEXT":
                all_header_types.append("text")
            else:
                all_header_types.append("number")
    row_content = list(map(list, zip(*vertical_content)))

    return {
        "header": all_headers,
        "rows": row_content,
        "types": all_header_types,
        "alias": list(_wtq_table_content["is_list"].keys())
    }


def main():
    dir_path = 'squall/tables'
    for table_file in tqdm(os.listdir(dir_path)):
        with open(os.path.join(dir_path, table_file)) as fp:
            table = json.load(fp)
        table = process_table_structure(table)
        with open(os.path.join(dir_path, table_file), 'w') as fp:
            json.dump(table, fp)


if __name__ == "__main__":
    main()
