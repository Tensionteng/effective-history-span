"""TSFM pretraining smoke runner: one run per invocation, logs stability metrics to CSV."""
import argparse, csv, os, sys, time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tsfm.model import TSFM
from tsfm.data import SynthTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hc_mode', default='res', choices=['res', 'hc', 'mhc', 'ohc'])
    ap.add_argument('--hc_expand', type=int, default=4)
    ap.add_argument('--layers', type=int, default=24)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--seq_len', type=int, default=512)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--tag', default='smoke')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = SynthTS(n_series=max(args.steps * args.batch, 50000), length=args.seq_len, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    drop_last=True, persistent_workers=True)

    model = TSFM(patch=args.patch, d_model=args.d_model, layers=args.layers, heads=args.heads,
                 max_len=args.seq_len, hc_mode=args.hc_mode, hc_expand=args.hc_expand).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    os.makedirs('logs/tsfm', exist_ok=True)
    csv_path = f'logs/tsfm/{args.tag}_{args.hc_mode}_L{args.layers}_lr{args.lr}.csv'
    t0 = time.time()
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'loss', 'grad_norm', 'act_amax', 'elapsed'])
        it = iter(dl)
        for step in range(1, args.steps + 1):
            try:
                x = next(it)
            except StopIteration:
                it = iter(dl)
                x = next(it)
            x = x.to(device, non_blocking=True)
            loss, z = model(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % args.log_every == 0 or step == 1:
                w.writerow([step, f'{loss.item():.5f}', f'{gn.item():.3f}',
                            f'{z.detach().abs().max().item():.1f}', f'{time.time() - t0:.1f}'])
                f.flush()
                print(f'step {step} loss {loss.item():.4f} gnorm {gn.item():.2f} amax {z.detach().abs().max().item():.1f}', flush=True)
    print(f'DONE params={n_params / 1e6:.1f}M csv={csv_path}')


if __name__ == '__main__':
    main()
