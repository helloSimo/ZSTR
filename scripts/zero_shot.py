import os
from argparse import ArgumentParser


def eval_model(q_name, p_name, dataset, device):
    encode_format_1 = "CUDA_VISIBLE_DEVICES={} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {} " \
                      "--per_device_eval_batch_size 64 " \
                      "--p_max_len 128 " \
                      "--encode_in_path datasets/{}_eval/corpus.jsonl " \
                      "--encoded_save_path zero-shot_temp/corpus_emb.pkl " \
                      "--dpr"
    encode_format_2 = "CUDA_VISIBLE_DEVICES={} python encode.py " \
                      "--output_dir temp_out " \
                      "--model_name_or_path {} " \
                      "--per_device_eval_batch_size 64 " \
                      "--q_max_len 32 " \
                      "--encode_in_path datasets/{}_eval/test.jsonl " \
                      "--encoded_save_path zero-shot_temp/test_emb.pkl " \
                      "--encode_is_qry " \
                      "--dpr"
    retrieve_shell = "python retrieve.py " \
                     "--query_reps zero-shot_temp/test_emb.pkl " \
                     "--passage_reps zero-shot_temp/corpus_emb.pkl " \
                     "--depth 100 " \
                     "--batch_size -1 " \
                     "--save_text " \
                     "--save_ranking_to zero-shot_temp/test_rank.csv"
    eval_format = "python eval.py " \
                  "--label_path datasets/{}_eval/label.jsonl " \
                  "--rank_path zero-shot_temp/test_rank.csv " \
                  "--result_txt"

    os.system(encode_format_1.format(device, p_name, dataset))
    os.system(encode_format_2.format(device, q_name, dataset))
    os.system(retrieve_shell)
    os.system(eval_format.format(dataset))


def eval_main(args):
    if not os.path.exists("zero-shot_temp"):
        os.mkdir("zero-shot_temp")

    dataset_list = [prefix+dataset for prefix in ['', 'D_'] for dataset in ['WTQ', 'WSQL', 'NQT', 'NQTt']]
    for dataset in dataset_list:
        eval_model(q_name=args.q_name,
                   p_name=args.p_name,
                   dataset=dataset,
                   device=args.device)

    with open("zero-shot_temp/result.txt") as f:
        for i, line in enumerate(f):
            print(dataset_list[i])
            print(line)

    os.system("rm -rf zero-shot_temp")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('q_name', type=str)
    parser.add_argument('p_name', type=str)
    parser.add_argument('device', type=int, choices=[0, 1, 2, 3])
    main_args = parser.parse_args()

    eval_main(main_args)
