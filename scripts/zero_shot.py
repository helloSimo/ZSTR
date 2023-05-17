import os
from argparse import ArgumentParser


def eval_model(q_name, p_name, dataset, device, dpr, result_dir):
    encode_format_1 = "CUDA_VISIBLE_DEVICES={0} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {1} " \
                      "--per_device_eval_batch_size 64 " \
                      "--p_max_len 128 " \
                      "--encode_in_path datasets/{2}_eval/corpus.jsonl " \
                      "--encoded_save_path {3}/corpus_emb.pkl "
    encode_format_2 = "CUDA_VISIBLE_DEVICES={0} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {1} " \
                      "--per_device_eval_batch_size 64 " \
                      "--q_max_len 32 " \
                      "--encode_in_path datasets/{2}_eval/test.jsonl " \
                      "--encoded_save_path {3}/test_emb.pkl " \
                      "--encode_is_qry "
    retrieve_shell = "python retrieve.py " \
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

    if dpr:
        encode_format_1 += "--dpr "
        encode_format_2 += "--dpr "

    os.system(encode_format_1.format(device, p_name, dataset, result_dir))
    os.system(encode_format_2.format(device, q_name, dataset, result_dir))
    os.system(retrieve_shell.format(result_dir))
    os.system(eval_format.format(dataset, result_dir))


def eval_main(args):
    result_dir = "{}_result".format(args.dataset)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    dataset_list = [prefix+dataset for prefix in ['', '3_wod_datasets/wod_'] for dataset in args.dataset.split(' ')]
    for dataset in dataset_list:
        eval_model(q_name=args.q_name,
                   p_name=args.p_name,
                   dataset=dataset,
                   device=args.device,
                   dpr=args.dpr,
                   result_dir=result_dir)

    with open(os.path.join(result_dir, "result.txt")) as f:
        for i, line in enumerate(f):
            print(dataset_list[i])
            print(line)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ')
    parser.add_argument('--q_name', type=str, default='Luyu/co-condenser-marco-retriever')
    parser.add_argument('--p_name', type=str, default='Luyu/co-condenser-marco-retriever')
    parser.add_argument('--device', type=int)
    parser.add_argument('--dpr', action='store_true')
    main_args = parser.parse_args()

    eval_main(main_args)
