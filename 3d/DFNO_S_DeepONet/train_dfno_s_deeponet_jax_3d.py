import sys, os, time, argparse, json
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax, jax.numpy as jnp, numpy as np, optax, equinox as eqx


class Tee:
    def __init__(self, fname, mode='w'):
        self.file = open(fname, mode)
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()


_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from model_dfno_s_deeponet_jax_3d import DFNOSeparableDeepONet3d, count_params

ntrain, ntest = 500, 100
batch_size = 25
learning_rate = 0.001
epochs = 2000
seed = 42
weight_decay = 1e-4

width, P, modes = 64, 25, 7
num_fourier_layers = 10

trunk_layers = [50, 50, 50, 50, 50, 50, 50]
R, nx, ny, nz = 50, 64, 64, 48

T_amb, delta_T, f_scale_branch = 298.15, 100.0, 0.03

data_dir = os.path.join(_this_dir, '..', '..', 'data', '3d')
base_save_path = os.path.join(_this_dir, 'results')


def rel_l2_norm(pred, true):
    num = jnp.linalg.norm((pred - true).ravel(), ord=2)
    den = jnp.linalg.norm(true.ravel(), ord=2)
    return num / (den + 1e-12)


def train(suffix='baseline'):
    device = 'GPU' if jax.default_backend() == 'gpu' else jax.default_backend().upper()
    model_name = 'D-FNO-BPSS'
    save_path = os.path.join(base_save_path, suffix)
    os.makedirs(save_path, exist_ok=True)
    tee = Tee(os.path.join(save_path, 'run_log.txt'), 'w')
    sys.stdout = tee
    np.random.seed(seed)

    print(f'JAX backend: {device}')
    print(f'Model: {model_name}')
    print(f'width={width}, P={P}, modes={modes}, layers={num_fourier_layers}')

    print('Loading data...')
    f_train = jnp.array(np.load(os.path.join(data_dir, 'f_train.npy')), dtype=jnp.float64)
    u_train = jnp.array(np.load(os.path.join(data_dir, 'u_train.npy')), dtype=jnp.float64)
    f_test = jnp.array(np.load(os.path.join(data_dir, 'f_test.npy')), dtype=jnp.float64)
    u_test = jnp.array(np.load(os.path.join(data_dir, 'u_test.npy')), dtype=jnp.float64)

    f_train = f_train[:ntrain] / f_scale_branch
    u_train = u_train[:ntrain]
    f_test = f_test[:ntest] / f_scale_branch
    u_test = u_test[:ntest]

    u_train = (u_train - T_amb) / delta_T
    u_test = (u_test - T_amb) / delta_T

    print(f'Train: {ntrain}, Test: {ntest}, Grid: {nx}x{ny}x{nz}, Batch: {batch_size}')

    print('Creating model...')
    key = jax.random.PRNGKey(0)
    cls = DFNOSeparableDeepONet3d
    model = cls(trunk_layers=trunk_layers, R=R, nx=nx, ny=ny, nz=nz,
                width=width, P=P, modes=modes,
                num_fourier_layers=num_fourier_layers, key=key)
    n_params = count_params(model)
    print(f'Parameters: {n_params:,}')

    schedule = optax.cosine_decay_schedule(learning_rate, epochs, 0.0)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, src, tgt):
        loss_val, grads = eqx.filter_value_and_grad(
            lambda m: jnp.mean((m(src) - tgt) ** 2))(model)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    @eqx.filter_jit
    def eval_forward(model, src):
        return model(src)

    def eval_all(model, src_all, tgt_all, eb=5):
        preds = []
        for start in range(0, src_all.shape[0], eb):
            end = min(start + eb, src_all.shape[0])
            preds.append(np.array(eval_forward(model, src_all[start:end])))
        pred_all = jnp.concatenate(preds, axis=0)
        pred_raw = pred_all * delta_T + T_amb
        tgt_raw = tgt_all * delta_T + T_amb
        diff = pred_raw - tgt_raw
        return (jnp.mean(diff ** 2), rel_l2_norm(pred_raw, tgt_raw),
                jnp.mean(jnp.abs(diff)), jnp.max(jnp.abs(diff)), pred_raw)

    loss_log = np.zeros(epochs, dtype=np.float64)
    test_rel_l2_log = np.zeros(epochs, dtype=np.float64)
    test_mae_log = np.zeros(epochs, dtype=np.float64)
    test_max_log = np.zeros(epochs, dtype=np.float64)
    total_time = 0.0
    log_every = max(1, epochs // 100)

    print(f'\n{"="*60}')
    print(f'Training {model_name} ({epochs} epochs)')
    print(f'{"="*60}\n')

    for ep in range(epochs):
        t1 = time.perf_counter()
        idx = np.random.choice(ntrain, batch_size, replace=False)
        model, opt_state, loss_val = train_step(model, opt_state,
                                                 f_train[idx], u_train[idx])

        if ep % log_every == 0 or ep == epochs - 1:
            _, rl2, mae, mx, _ = eval_all(model, f_test, u_test)
            trv, tmv, txv = rl2.item(), mae.item(), mx.item()
        else:
            trv = test_rel_l2_log[max(0, ep - 1)] if ep > 0 else 1.0
            tmv = test_mae_log[max(0, ep - 1)] if ep > 0 else 0.0
            txv = test_max_log[max(0, ep - 1)] if ep > 0 else 0.0

        t2 = time.perf_counter()
        total_time += (t2 - t1)

        loss_log[ep] = float(loss_val)
        test_rel_l2_log[ep] = trv
        test_mae_log[ep] = tmv
        test_max_log[ep] = txv

        if ep % log_every == 0:
            print(f'Iter {ep:5d} | Time {t2-t1:.2f}s | '
                  f'Loss {loss_val:.2e} | '
                  f'Test RL2 {trv:.6f} MAE {tmv:.4f}K Max {txv:.4f}K')

    print(f'\n{"="*60}')
    print(f'Training complete. Total time: {total_time:.1f}s')
    print(f'Final Test Rel.L2: {test_rel_l2_log[-1]:.6f} ({100*test_rel_l2_log[-1]:.4f}%)')
    print(f'Final MAE: {test_mae_log[-1]:.4f}K  MaxAbs: {test_max_log[-1]:.4f}K')
    print(f'{"="*60}\n')

    print('Computing per-sample test error...')
    _, _, mae, mx, pred_all = eval_all(model, f_test, u_test)
    pred_np = np.array(pred_all)
    u_test_raw = np.load(os.path.join(data_dir, 'u_test.npy'))
    u_test_np = np.array(u_test_raw)[:ntest]
    errors = np.zeros(len(u_test_np), dtype=np.float64)
    for i in range(len(u_test_np)):
        errors[i] = rel_l2_norm(jnp.array(pred_np[i]), jnp.array(u_test_np[i])).item()
    print(f'Mean Testing Error: {100*errors.mean():.4f}%')
    print(f'MAE: {mae.item():.4f}K  MaxAbs: {mx.item():.4f}K')
    print(f'Total Time: {total_time:.1f}s  Params: {n_params:,}')

    eqx.tree_serialise_leaves(os.path.join(save_path, 'dfno_deeponet_3d.eqx'), model)
    np.savez(os.path.join(save_path, 'train_metrics.npz'),
             train_loss=loss_log, test_rel_l2=test_rel_l2_log,
             test_mae=test_mae_log, test_max=test_max_log)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(dict(ntrain=ntrain, batch_size=batch_size,
                       epochs=epochs, lr=learning_rate,
                       T_amb=T_amb, delta_T=delta_T,
                       width=width, P=P, modes=modes, R=R,
                       trunk_layers=trunk_layers,
                       num_fourier_layers=num_fourier_layers,
                       nx=nx, ny=ny, nz=nz), f, indent=2)

    sys.stdout = tee.stdout
    tee.close()
    print(f'Results saved to {save_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='baseline')
    args = parser.parse_args()
    train(suffix=args.suffix)
