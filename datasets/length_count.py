import jsonlines
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer


def fix_length(length):
    if length < 32:
        return 32
    elif length < 64:
        return 64
    elif length < 128:
        return 128
    elif length < 256:
        return 256
    else:
        return 512


def print_result(count, tot):
    for i in [32, 64, 128, 256, 512]:
        print('{:<}\t'.format(i), end='')
    print()
    c = 0
    for i in [32, 64, 128, 256, 512]:
        c += count[i]
        print('{:<2.2f}\t'.format(c / tot * 100), end='')
    print()


#tokenizer = AutoTokenizer.from_pretrained("tokenizer")
tokenizer = None

q_count = OrderedDict()
p_count = OrderedDict()
for i in [32, 64, 128, 256, 512]:
    q_count[i] = 0
    p_count[i] = 0

q_tot = 0
p_tot = 0
for data in tqdm(jsonlines.open('train/train.jsonl', 1)):
    q_length = len(tokenizer.tokenize(data['query']))
    q_length = fix_length(q_length)
    q_count[q_length] += 1
    q_tot += 1

    for name in ['positives', 'negatives']:
        for p in data[name]:
            p_length = len(tokenizer.tokenize(p))
            p_length = fix_length(p_length)
            p_count[p_length] += 1
            p_tot += 1

print('query length:')
print_result(q_count, q_tot)

print('passage length:')
print_result(p_count, p_tot)
