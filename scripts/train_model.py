import os
from argparse import ArgumentParser


def train_model(device, dataset, max_len, lr_name, epoch_num):
    shell_format = "CUDA_VISIBLE_DEVICES={0} python train.py " \
                   "--untie_encoder --do_train  --do_eval  --evaluation_strategy epoch " \
                   "--model_name_or_path vblagoje/dpr-question_encoder-single-lfqa-wiki " \
                   "--additional_model_name vblagoje/dpr-ctx_encoder-single-lfqa-wiki " \
                   "--per_device_train_batch_size 64  --per_device_eval_batch_size 256 " \
                   "--gradient_accumulation_steps 1  --warmup_ratio  0.1 --fp16 --dpr " \
                   "--logging_first_step  --logging_strategy  epoch  --save_strategy  epoch " \
                   "--save_total_limit 4  --early_stop 0  --dataloader_num_workers 2 " \
                   "--load_best_model_at_end  --metric_for_best_model eval_acc " \
                   "--train_negative_num 1  --dev_negative_num 8 --q_max_len 32 " \
                   "--train_dir datasets/{1}_train " \
                   "--output_dir model_dpr_{1}_{2}_{3}_{5} " \
                   "--p_max_len {2} " \
                   "--learning_rate {4} " \
                   "--num_train_epochs {5} " \
                   "--eval_delay {6}"

    model_name = "model_dpr_{}_{}_{}_{}".format(dataset, max_len, lr_name, epoch_num)
    if os.path.exists(model_name):
        if os.path.exists(os.path.join(model_name, 'setting.txt')):
            return
        else:
            os.system('rm -rf {}'.format(model_name))

    lr = lr_name.replace('e', 'e-')
    delay_num = epoch_num-40 if epoch_num > 40 else 20
    shell = shell_format.format(device, dataset, max_len, lr_name, lr, epoch_num, delay_num)
    os.system(shell)


def train(args):
    if args.dataset in ['WTQ', 'WSQL']:
        assert args.epoch_num in [40, 80]
    else:
        assert args.epoch_num in [120, 160]

    train_model(device=args.device,
                dataset=args.prefix+args.dataset,
                max_len=args.length,
                lr_name=args.lr_name,
                epoch_num=args.epoch_num)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--device', type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--dataset', type=str, choices=['WTQ', 'WSQL', 'NQT', 'NQTt'])
    parser.add_argument('--length', type=int, choices=[128, 256])
    parser.add_argument('--lr_name', type=str, choices=['2e5', '1e5', '5e6', '1e6'])
    parser.add_argument('--epoch_num', type=int)

    main_args = parser.parse_args()

    train(main_args)
