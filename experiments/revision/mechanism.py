"""Long-horizon controlled hierarchy experiments with total curvature gradients.

All cases, fixed-K controls, and pair splits are declared before execution.
These are synthetic mechanism checks, not additional real-data task families.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from geometry import Geometry, L


def schedule(cfg, epoch):
    groups = len(cfg['branchings'])
    end = 1 + (cfg['scale_end'] - 1) * np.arange(1, groups + 1) / groups
    if cfg['schedule'] == 'abrupt':
        alpha = float(epoch >= cfg['epochs'] // 2)
    elif cfg['schedule'] == 'smooth':
        alpha = epoch / max(1, cfg['epochs'] - 1)
    else:
        alpha = 0.
    return 1 + alpha * (end - 1)


def run(cfg, root, device, out):
    from train_multigroup import gen_subtree, graph_dist
    out = Path(out)
    n = cfg['nodes_per_group']
    groups = len(cfg['branchings'])
    distances = []
    for g, branching in enumerate(cfg['branchings']):
        edges, count = gen_subtree(branching, n, 42 + g)
        assert count == n
        distances.append(graph_dist(n, edges, device))
    distances = torch.stack(distances)
    pairs = torch.triu_indices(n, n, 1, device=device)
    # The pair split is identical for all methods and is independent of initialization.
    order = torch.randperm(pairs.shape[1], generator=torch.Generator().manual_seed(2042)).to(device)
    cuts = [int(.8 * len(order)), int(.9 * len(order))]
    split = dict(zip(['train', 'validation', 'test'], torch.tensor_split(order, cuts)))
    partitions = {key: pairs[:, ids] for key, ids in split.items()}
    geo = Geometry([(n, cfg['dim'])] * groups, cfg['method'], device,
                   lr=cfg['lr'], rho=cfg['rho'], beta=cfg['beta'], gamma=1.,
                   radius=4., init_K=cfg.get('init_K', 1.), k_min=.001)
    start = time.monotonic()
    history = []
    metadata = dict(protocol='sum of within-group relative squared-distance losses',
                    total_nodes=n * groups, groups=groups, dimension_per_group=cfg['dim'],
                    split_seed=2042, pair_counts={k:len(v) for k,v in split.items()},
                    precision='FP32', curvature_derivative='total derivative through point homothety',
                    curvature_interval=[.001,5.], canonical_radius_cap=4.,
                    selection='No outcome-based early stopping; report final test and complete validation AUC',
                    role='Synthetic mechanism diagnostic, separate from product-factor benchmarks')
    (out / 'protocol.json').write_text(json.dumps(metadata, indent=2))

    def errors(indices, scale):
        i, j = indices
        ps = torch.stack(geo.chart_points())
        ds = L.dist(ps[:, i], ps[:, j], geo.K()[:, None])
        return ds / (distances[:, i, j] * scale[:, None]) - 1

    permanent = {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
    for epoch in range(cfg['epochs']):
        geo.epoch = epoch
        scale = torch.tensor(schedule(cfg, epoch), device=device, dtype=torch.float32)
        geo.zero_grad()
        err = errors(partitions['train'], scale)
        group_loss = err.square().mean(-1)
        loss = group_loss.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite mechanism loss at epoch {epoch}')
        loss.backward()
        curvature_gradient = geo.c.grad.detach().clone() if geo.c.grad is not None else torch.zeros_like(geo.c)
        old_k = geo.K().detach().clone()
        geo.step()
        with torch.no_grad():
            val = errors(partitions['validation'], scale).abs().mean(-1)
            k = geo.K().detach()
            if not torch.isfinite(val).all() or not torch.isfinite(k).all():
                raise FloatingPointError(f'Nonfinite mechanism evaluation at epoch {epoch}')
            constraint64=[]; transport_errors=[]
            for g,p in enumerate(geo.points):
                constraint64.append(float((k[g].double()*L.lorentz_inner(p.double(),p.double())+1).abs().max()))
                transported=L.parallel_transport(p[:1],p[1:2],geo.m[g][:1],k[g])
                norm=L.lorentz_norm(geo.m[g][:1])
                transport_errors.append(float((L.lorentz_norm(transported)-norm).abs().max()))
            row = dict(epoch=epoch, train_loss=float(group_loss.mean()),
                       train_distortion=float(err.detach().abs().mean()),
                       val_distortion=float(val.mean()), val_distortion_groups=val.cpu().tolist(),
                       K=k.cpu().tolist(), scale=scale.cpu().tolist(),
                       curvature_gradient=curvature_gradient.cpu().tolist(),
                       normalized_constraint_fp64=max(constraint64),
                       parallel_transport_norm_error=max(transport_errors),
                       log_curvature_step=(k/old_k).log().cpu().tolist(),
                       seconds=time.monotonic()-start,
                       peak_memory_mb=torch.cuda.max_memory_allocated()/1e6, **geo.last)
            if epoch == cfg['epochs'] - 1:
                test = errors(partitions['test'], scale).abs().mean(-1)
                row.update(test_distortion=float(test.mean()), test_distortion_groups=test.cpu().tolist())
        history.append(row)
        with (out/'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if epoch in permanent or (epoch+1)%100==0:
            state = dict(epoch=epoch, config=cfg, geometry=geo.state_dict(),
                         momentum=[m.detach().cpu() for m in geo.m], rng=torch.get_rng_state())
            torch.save(state,out/f'checkpoint_{epoch:04d}.pt')
            torch.save(state,out/'last.pt')
        if epoch%100==0 or epoch==cfg['epochs']-1:
            print(json.dumps(row),flush=True)
    values=np.array([r['val_distortion'] for r in history])
    final=dict(history[-1],validation_auc=float(values.mean()),
               validation_auc_first_half=float(values[:len(values)//2].mean()),
               validation_auc_second_half=float(values[len(values)//2:].mean()),
               final_K_spread=float(np.std(history[-1]['K'])),
               max_manifold_residual=max(r['manifold_residual'] for r in history),
               max_tangent_residual=max(r['tangent_residual'] for r in history))
    (out/'result.json').write_text(json.dumps(dict(status='complete',config=cfg,
        epochs_completed=len(history),final=final,protocol=metadata),indent=2,allow_nan=False))
