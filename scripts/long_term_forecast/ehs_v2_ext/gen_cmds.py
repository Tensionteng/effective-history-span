#!/usr/bin/env python
"""Generate command lists for the EHS v2 EXTENSION (new datasets) lookback study.

Same protocol as scripts/long_term_forecast/ehs_v2/gen_cmds.py:
  new datasets x {iTransformer, DLinear} x seq_len {96,336,720,1440,2880}
  x seeds {2021,2022,2023}, pred_len 96,
  all arms with --max_train_windows = N_min(sl=2880) of that dataset.

Config rule (task spec): wide data (>100 channels) uses the electricity config
(dm512/df512/el3/lr5e-4/bs16); narrow data uses the ETT config
(dm128/df128/el2/lr1e-4/bs32).

Outputs: main_cmds.txt (one run per line, GPU id baked in round-robin).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_LENS = [96, 336, 720, 1440, 2880]
SEEDS = [2021, 2022, 2023]

# name: (data_arg, root, csv, enc_in, d_model, d_ff, e_layers, lr, bs)
def wide(root, csv, enc):
    return ('custom', root, csv, enc, 512, 512, 3, '0.0005', 16)


def narrow(root, csv, enc):
    return ('custom', root, csv, enc, 128, 128, 2, '0.0001', 32)


FAMILY = {
    'solar': wide('./dataset/solar/', 'solar.csv', 137),
    'PEMS04': wide('./dataset/PEMS04/', 'PEMS04.csv', 307),
    'PEMS08': wide('./dataset/PEMS08/', 'PEMS08.csv', 170),
    'ercot': narrow('./dataset/ercot/', 'ercot.csv', 8),
    'pedestrian': narrow('./dataset/pedestrian/', 'pedestrian.csv', 13),
}

# N_min = train windows of the seq_len=2880 arm (pred_len=96):
# N_min = int(n_rows * 0.7) - 2880 - 96 + 1   (Dataset_Custom split)
N_MIN = {
    'solar': 33817,      # 52560 rows, num_train=36792
    'PEMS04': 8919,      # 16992 rows, num_train=11894
    'PEMS08': 9524,      # 17856 rows, num_train=12499
    'ercot': 105435,     # 154872 rows, num_train=108410
    'pedestrian': 56056,  # 84331 rows, num_train=59031
}


def cmd(name, model, sl, seed, nmin, log, gpu):
    data, root, csv, enc, dm, df, el, lr, bs = FAMILY[name]
    return (f'CUDA_VISIBLE_DEVICES={gpu} .venv/bin/python -u run.py '
            f'--task_name long_term_forecast --is_training 1 '
            f'--root_path {root} --data_path {csv} '
            f'--model_id LBext_{name}_{model}_{sl}_s{seed} --model {model} '
            f'--data {data} --features M --seq_len {sl} --label_len 48 --pred_len 96 '
            f'--e_layers {el} --d_layers 1 --factor 3 --enc_in {enc} --dec_in {enc} --c_out {enc} '
            f'--des Exp --d_model {dm} --d_ff {df} --n_heads 8 '
            f'--learning_rate {lr} --batch_size {bs} --itr 1 --seed {seed} '
            f'--num_workers 4 '
            f'--max_train_windows {nmin} > {log} 2>&1')


def main():
    assert N_MIN, 'fill N_MIN after dataset prep'
    lines = []
    os.makedirs('logs/ehs_v2_ext', exist_ok=True)
    i = 0
    for seed in SEEDS:
        for sl in SEQ_LENS:
            for model in ['iTransformer', 'DLinear']:
                for name in FAMILY:
                    log = f'logs/ehs_v2_ext/{name}_{model}_sl{sl}_s{seed}.log'
                    lines.append(cmd(name, model, sl, seed, N_MIN[name], log, i % 8))
                    i += 1
    with open(os.path.join(HERE, 'main_cmds.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'main_cmds.txt: {len(lines)} runs')


if __name__ == '__main__':
    main()
