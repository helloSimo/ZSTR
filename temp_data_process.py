import os
import jsonlines


def process_table(table):
    text = ' '.join(table['header'])
    for row in table['rows']:
        text += ' ' + ' '.join(row)
    return text


def process_corpus(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')

    dest_f = jsonlines.open('data/corpus.jsonl', 'w')
    for table in jsonlines.open(f'datasets/{dataset}/tables.jsonl'):
        dest_f.write({
            'docid': table['id'],
            'title': table['title'],
            'text': process_table(table)
        })


def main():
    dataset = "WTQ"
    process_corpus(dataset)


if __name__ == "__main__":
    main()