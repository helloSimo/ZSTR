import os
from argparse import ArgumentParser


def name_model(dataset, max_len, lr_name, epoch_num, model_name):
    return "model_{}_{}_{}_{}_{}".format(model_name, dataset, max_len, lr_name, epoch_num)


def train_model(device, dataset, max_len, lr_name, epoch_num, dpr, untie_encoder, model, additional_model, model_name):
    shell_format = "CUDA_VISIBLE_DEVICES={0} python train.py " \
                   "--do_train  --do_eval  --evaluation_strategy epoch " \
                   "--model_name_or_path {7} " \
                   "--per_device_train_batch_size 128  --per_device_eval_batch_size 256 " \
                   "--gradient_accumulation_steps 1  --warmup_ratio  0.1 --fp16 " \
                   "--logging_first_step  --logging_strategy  epoch  --save_strategy  epoch " \
                   "--save_total_limit 4  --early_stop 0  --dataloader_num_workers 2 " \
                   "--metric_for_best_model eval_acc " \
                   "--train_negative_num 1  --dev_negative_num 8 --q_max_len 32 " \
                   "--train_dir datasets/{1}_train " \
                   "--output_dir {8} " \
                   "--p_max_len {2} " \
                   "--learning_rate {4} " \
                   "--num_train_epochs {5} " \
                   "--eval_delay {6} "
    if dpr:
        shell_format += "--dpr "
    if untie_encoder:
        shell_format += "--untie_encoder --additional_model_name {0} ".format(additional_model)

    output_dir = name_model(dataset, max_len, lr_name, epoch_num, model_name)
    if os.path.exists(output_dir):
        if os.path.exists(os.path.join(output_dir, 'setting.txt')):
            return
        else:
            os.system('rm -rf {}'.format(output_dir))

    lr = lr_name.replace('e', 'e-')
    delay_num = 0
    shell = shell_format.format(device, dataset, max_len, lr_name, lr, epoch_num, delay_num, model, output_dir)
    os.system(shell)


def train(args):
    if args.dataset in ['WTQ', 'WSQL']:
        assert args.epoch_num in [20, 40]
    else:
        assert args.epoch_num in [80, 120, 160]

    train_model(device=args.device,
                dataset=args.prefix+args.dataset,
                max_len=args.length,
                lr_name=args.lr_name,
                epoch_num=args.epoch_num,
                dpr=args.dpr,
                untie_encoder=args.untie_encoder,
                model=args.model,
                additional_model=args.additional_model,
                model_name=args.model_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--device', type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--model_name', type=str, default='condenser')
    parser.add_argument('--model', type=str, default='Luyu/co-condenser-wiki')
    parser.add_argument('--additional_model', type=str, default='Luyu/co-condenser-wiki')
    parser.add_argument('--dataset', type=str, choices=['WTQ', 'WSQL', 'NQT', 'NQTt'])
    parser.add_argument('--length', type=int, choices=[128, 256])
    parser.add_argument('--lr_name', type=str, choices=['2e5', '1e5', '5e6', '1e6'])
    parser.add_argument('--epoch_num', type=int)
    parser.add_argument('--dpr', action='store_true')
    parser.add_argument('--untie_encoder', action='store_true')

    main_args = parser.parse_args()

    train(main_args)
