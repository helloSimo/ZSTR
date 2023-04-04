import os
from argparse import ArgumentParser


def get_model_list(lr_name):
    model_list = []
    for dir_name in os.listdir('..'):
        if dir_name.startswith('model_dpr') and lr_name in dir_name:
            model_list.append(dir_name)
    return model_list


def get_step_list(model_name):
    step_list = []
    for dir_name in os.listdir(model_name):
        if dir_name.startswith("checkpoint"):
            step_list.append(int(dir_name.split('-')[-1]))
    return sorted(step_list)


def get_epoch_list(model_name):
    step_list = get_step_list(model_name)
    max_epoch = int(model_name.split('_')[-1])
    min_epoch = max_epoch - len(step_list) + 2
    best_epoch = int(step_list[0] / step_list[-1] * max_epoch)

    epoch_list = [best_epoch]
    for i in range(len(step_list)-1):
        epoch_list.append(min_epoch+i)
    return epoch_list


def get_best_ckpt(model_name, print_result=True):
    with open(os.path.join(model_name, 'result.txt')) as f:
        print(f.read())

    results = []
    lines = []
    with open(os.path.join(model_name, 'result.txt')) as f:
        for i, line in enumerate(f):
            lines.append(line)

            rs = []
            for result in line.split('\t'):
                rs.append(float(result))
            results.append((i, (sum(rs[1:4]), sum(rs))))

    max_id = max(results, key=lambda x: x[1])[0]
    epoch_list = get_epoch_list(model_name)
    if print_result:
        print("best epoch:{}".format(epoch_list[max_id]))
        print(lines[max_id])

    with open(os.path.join(model_name, 'result.txt'), 'a+') as f:
        print("best epoch:{}".format(epoch_list[max_id]), file=f)
        print(lines[max_id], file=f)

    return epoch_list[max_id], lines[max_id]


def eval_model(model_name, device):
    encode_format_1 = "CUDA_VISIBLE_DEVICES={4} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {0}/checkpoint-{1} " \
                      "--per_device_eval_batch_size 1024 " \
                      "--p_max_len {2} " \
                      "--encode_in_path datasets/{3}_eval/corpus.jsonl " \
                      "--encoded_save_path {0}/corpus_emb.pkl "
    encode_format_2 = "CUDA_VISIBLE_DEVICES={3} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {0}/checkpoint-{1} " \
                      "--per_device_eval_batch_size 1024 " \
                      "--q_max_len 32 " \
                      "--encode_in_path datasets/{2}_eval/test.jsonl " \
                      "--encoded_save_path {0}/test_emb.pkl " \
                      "--encode_is_qry "
    retrieve_format = "python retrieve.py " \
                      "--query_reps {0}/test_emb.pkl " \
                      "--passage_reps {0}/corpus_emb.pkl " \
                      "--depth 100 " \
                      "--batch_size -1 " \
                      "--save_text " \
                      "--save_ranking_to {0}/test_rank.csv"
    eval_format = "python eval.py " \
                  "--label_path datasets/{0}_eval/label.jsonl " \
                  "--rank_path {1}/test_rank.csv " \
                  "--result_txt"

    param_list = model_name.split('_')
    dataset_name = param_list[-4]
    p_max_len = param_list[-3]

    for ckpt in get_step_list(model_name):
        os.system(encode_format_1.format(model_name, ckpt, p_max_len, dataset_name, device))
        os.system(encode_format_2.format(model_name, ckpt, dataset_name, device))
        os.system(retrieve_format.format(model_name))
        os.system(eval_format.format(dataset_name, model_name))


def eval_main(args):
    model_name = args.model_name
    device = args.device

    eval_model(model_name, device)
    get_best_ckpt(model_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('device', type=int)
    main_args = parser.parse_args()

    eval_main(main_args)
