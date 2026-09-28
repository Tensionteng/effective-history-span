"""Train one arm of the latent-action probe: --arm {baseline,cont,fsq}.

Next-6-patch MSE on the shock synthetic set, AdamW lr 1e-3 + cosine, batch 64.
Saves checkpoint + train log to logs/latent_action/, then runs eval_action.
"""
import argparse, csv, math, os, sys, time, json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tsfm.shock_data import ShockTS
from tsfm.latent_action import LatentActionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['baseline', 'cont', 'fsq'])
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--code_smooth', type=float, default=0.0,
                    help='weight of temporal smoothness loss on (quantized) action latent')
    ap.add_argument('--code_smooth_mode', default='l2', choices=['l2', 'l1'],
                    help='l2: mean sq. change; l1: mean L2-norm of change (total-variation, '
                         'promotes exactly-constant segments with free jump size)')
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--dec_layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--loss_hmin', type=int, default=1,
                    help='train loss only on horizon steps hmin..H (1 = all)')
    ap.add_argument('--seq_len', type=int, default=512)
    ap.add_argument('--latent_dim', type=int, default=3)
    ap.add_argument('--shift_res', action='store_true',
                    help='action bottleneck: decoder sees raw state up to patch i-1, '
                         'patch i only via the action latent')
    ap.add_argument('--action_input', default='state', choices=['state', 'innov', 'diff'],
                    help='action head input: encoder state (v1) / predictor residual / patch diff')
    ap.add_argument('--null_target', type=float, default=0.0,
                    help='>0: reserve code 0 as null, force positions below this innovation-norm '
                         'quantile to null (sparse firing)')
    ap.add_argument('--gate_mode', default='abs', choices=['abs', 'white'],
                    help='white: null gate fires on scale-whitened innovation (surprise)')
    ap.add_argument('--gate_thresh', type=float, default=0.0,
                    help='>0: absolute surprise threshold (c x EMA-RMS of whitened norm), '
                         'firing rate emerges from data; overrides null_target quantile')
    ap.add_argument('--pred_mlp', action='store_true', help='2-layer innovation predictor')
    ap.add_argument('--code_feat', default='raw', choices=['raw', 'rich'],
                    help='rich: code input = [raw, whitened, 4-step group norms] (needs white gate)')
    ap.add_argument('--gate_accum', action='store_true',
                    help='gate also fires on 0.8*||white_i + white_{i-1}|| (accumulated drift)')
    ap.add_argument('--loss_type', default='mse', choices=['mse', 'nll'])
    ap.add_argument('--aux_weight', type=float, default=1.0,
                    help='weight of the auxiliary innovation-predictor loss')
    ap.add_argument('--fsq_levels', type=int, nargs='+', default=[5, 5, 5])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise_lo', type=float, default=0.2)
    ap.add_argument('--noise_hi', type=float, default=0.8,
                    help='AR innovation std upper range; lower = quieter regime')
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out_dir', default='logs/latent_action')
    ap.add_argument('--no_eval', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    lv = ''.join(str(l) for l in args.fsq_levels)
    name = (args.arm + ('_bott' if args.shift_res else '')
            + (f'_l{lv}' if (args.arm == 'fsq' and args.fsq_levels != [5, 5, 5]) else '')
            + (f'_hm{args.loss_hmin}' if args.loss_hmin > 1 else '')
            + (f'_q{args.noise_hi}' if args.noise_hi != 0.8 else '')
            + (f'_{args.action_input}' if args.action_input != 'state' else '')
            + (f'_null{args.null_target}' if args.null_target > 0 else '')
            + ('_gw' if args.gate_mode == 'white' else '')
            + (f'_gt{args.gate_thresh}' if args.gate_thresh > 0 else '')
            + ('_pm' if args.pred_mlp else '')
            + ('_rich' if args.code_feat == 'rich' else '')
            + ('_acc' if args.gate_accum else '')
            + ('_nll' if args.loss_type == 'nll' else '')
            + ((f'_sm{args.code_smooth}' if args.code_smooth_mode == 'l2' else f'_tv{args.code_smooth}')
               if args.code_smooth > 0 else '') + args.tag)

    ds = ShockTS(n_series=args.steps * args.batch, length=args.seq_len, seed=args.seed,
                 noise_lo=args.noise_lo, noise_hi=args.noise_hi)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

    model = LatentActionModel(arm=args.arm, patch=args.patch, d_model=args.d_model,
                              layers=args.layers, heads=args.heads, dec_layers=args.dec_layers,
                              horizon=args.horizon, max_len=args.seq_len,
                              fsq_levels=tuple(args.fsq_levels), latent_dim=args.latent_dim,
                              shift_res=args.shift_res, action_input=args.action_input,
                              null_target=args.null_target, loss_type=args.loss_type,
                              gate_mode=args.gate_mode, gate_thresh=args.gate_thresh,
                              pred_mlp=args.pred_mlp, code_feat=args.code_feat,
                              gate_accum=args.gate_accum).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_mult(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        t = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    csv_path = os.path.join(args.out_dir, f'train_{name}.csv')
    t0 = time.time()
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'mse', 'smooth', 'lr', 'elapsed'])
        for step, batch in enumerate(dl, 1):
            x = batch['x'].to(device, non_blocking=True)
            pred, codes, q, p, logvar, aux = model(x)
            mse, _ = model.multi_patch_loss(pred, p, logvar=logvar, hmin=args.loss_hmin)
            if q is not None and args.code_smooth > 0:
                dq = q[:, 1:] - q[:, :-1]
                smooth = dq.pow(2).mean() if args.code_smooth_mode == 'l2' else dq.norm(dim=-1).mean()
            else:
                smooth = None
            loss = mse + (args.code_smooth * smooth if smooth is not None else 0.0)
            if aux is not None:
                loss = loss + args.aux_weight * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0 or step == 1:
                w.writerow([step, f'{mse.item():.5f}',
                            f'{smooth.item():.5f}' if smooth is not None else '',
                            f'{sched.get_last_lr()[0]:.2e}', f'{time.time() - t0:.1f}'])
                f.flush()
                print(f'[{name}] step {step}/{args.steps} mse {mse.item():.4f}'
                      + (f' smooth {smooth.item():.4f}' if smooth is not None else '')
                      + f' ({time.time() - t0:.0f}s)', flush=True)
            if step >= args.steps:
                break

    ckpt_path = os.path.join(args.out_dir, f'ckpt_{name}.pt')
    torch.save({'model': model.state_dict(), 'args': vars(args), 'params': n_params}, ckpt_path)
    print(f'DONE {name} params={n_params / 1e6:.2f}M ckpt={ckpt_path} ({time.time() - t0:.0f}s)')

    if not args.no_eval:
        from tsfm.eval_action import evaluate
        metrics = evaluate(ckpt_path, device=device, out_dir=args.out_dir, name=name)
        print(json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
