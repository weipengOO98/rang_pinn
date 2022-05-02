"""

"""
import torch
from tools import gradients, MLP, logger
import matplotlib.pyplot as plt
import numpy as np
from ff import error_ff, hammersely, lhs
from parser_pinn import get_parser
import pathlib

parser_PINN = get_parser()
args = parser_PINN.parse_args()
path = pathlib.Path(args.save_path)
path.mkdir(exist_ok=True, parents=True)
for key, val in vars(args).items():
    print(f"{key} = {val}")
with open(path.joinpath('config'), 'wt') as f:
    f.writelines([f"{key} = {val}\n" for key, val in vars(args).items()])
maxiter: int = int(args.maxiter)
net_seq: list = list(args.net_seq)
sample_num: int = int(args.sample_num)
resample_interval: int = int(args.resample_interval)
freq_draw: int = int(args.freq_draw)
verbose: bool = bool(args.verbose)
resample_N: int = int(args.resample_N)
t_start = float(args.t_start)
alpha = float(args.alpha)
Tmax= float(args.Tmax)
half_x = float(args.half_L)

resample_num = maxiter // resample_interval
log_interval = maxiter // 10
rar_interval = maxiter // resample_num

# todo more careful check
GPU_ENABLED = True
if torch.cuda.is_available():
    try:
        _ = torch.Tensor([0., 0.]).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('gpu available')
        GPU_ENABLED = True
    except:
        print('gpu not available')
        GPU_ENABLED = False
else:
    print('gpu not available')
    GPU_ENABLED = False

_memo = []


def exact_heat():
    if len(_memo) == 0:
        xc = np.linspace(-1, 1, 501)*half_x
        tc = np.linspace(0, 1, 501)*Tmax
        xx, tt = np.meshgrid(xc, tc)
        u = 0.1 / np.sqrt(alpha * (tt + t_start)) * np.exp(-(xx+2-4*tt) ** 2 / (4 * alpha * (tt + t_start)))
        _memo.append((xx.reshape(-1, 1), tt.reshape(-1, 1), u.reshape(-1, 1)))
    return _memo[0]


def compute_lhs(u, x, t):
    u_pred = u(torch.cat([x, t], dim=1))
    u_x = gradients(u_pred, x, 1)
    u_xx = gradients(u_x, x, 1)
    u_t = gradients(u_pred, t, 1)
    lhs = u_t + 4*u_x - alpha*u_xx
    return lhs


def compute_res(u):
    resample_N = 500
    xc = torch.linspace(-1, 1, resample_N)*half_x
    tc = torch.linspace(0, Tmax, resample_N)

    xx, yy = torch.meshgrid(xc, tc, indexing='xy')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)

    x = torch.Tensor(xy[:, 0]).reshape(-1, 1).requires_grad_(True)
    t = torch.Tensor(xy[:, 1]).reshape(-1, 1).requires_grad_(True)

    lhs = compute_lhs(u, x, t)
    residual = torch.abs(lhs)
    error = residual.reshape(resample_N, resample_N).detach().cpu().numpy()

    xc = np.linspace(-1, 1, resample_N)
    tc = np.linspace(-1, 1, resample_N)
    xx, yy = np.meshgrid(xc, tc, indexing='xy')

    return xx, yy, error


def ddu_fn(x, y):
    return torch.zeros_like(x)


def l2_loss(u):
    xx, tt, u_truth = exact_heat()
    xx = torch.Tensor(xx)
    tt = torch.Tensor(tt)
    u_truth = torch.Tensor(u_truth)
    xy = torch.cat([xx, tt], dim=1)
    with torch.no_grad():
        u_pred = u(xy)
        l2_error = torch.sqrt(
            torch.sum((u_pred - u_truth) ** 2) / torch.sum((u_truth ** 2)))
    return l2_error.detach().cpu().numpy()


def several_error(u):
    with torch.no_grad():
        xx, tt, u_truth = exact_heat()
        xx = torch.Tensor(xx)
        tt = torch.Tensor(tt)
        u_truth = torch.Tensor(u_truth)
        xy = torch.cat([xx, tt], dim=1)
        u_pred = u(xy)
        mse = torch.mean((u_pred - u_truth) ** 2).detach().cpu().numpy()

    return mse


torch.random.seed()


def interior(n=sample_num, method='random'):
    if method == 'random':
        xx = (torch.rand(n, 1) * 2 - 1.)*half_x
        yy = torch.rand(n, 1)*Tmax
    elif method == 'uniform':
        N = int(np.sqrt(n))
        xc = torch.linspace(-1, 1, N)*half_x
        tc = torch.linspace(0, 1, N)*Tmax
        xx, yy = torch.meshgrid(xc, tc)
        xx = xx.ravel().reshape(-1, 1)
        yy = yy.ravel().reshape(-1, 1)
    elif method == "hammersely":
        xy = hammersely(n)
        xx, yy = torch.Tensor(xy[:, 0:1] * 2 - 1)*half_x, torch.Tensor(xy[:, 1:2])*Tmax
    elif method == "lhs":
        xy = lhs(n)
        xx, yy = torch.Tensor(xy[:, 0:1] * 2 - 1)*half_x, torch.Tensor(xy[:, 1:2])*Tmax
    cond = ddu_fn(xx, yy)
    return xx.requires_grad_(True), yy.requires_grad_(True), cond


def interior_ff(error, n=sample_num, verbose=False):
    xy = error_ff(n,
                  error=error,
                  max_min_density_ratio=10,
                  box=[0, 1, 0, 1])
    if verbose:
        logger.info("length of samples is {}".format(len(xy)))
    # logger.info(f"FF sample number is {len(xy)}")
    x = torch.Tensor(xy[:, 0] * 2 - 1).reshape(-1, 1)*half_x
    y = torch.Tensor(xy[:, 1]).reshape(-1, 1)*Tmax
    cond = ddu_fn(x, y)

    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    return x, y, cond


def boundary(n=100):
    xl = -torch.ones(n).reshape(-1, 1)*half_x
    xr = torch.ones(n).reshape(-1, 1)*half_x
    tl = torch.linspace(0, Tmax, n).reshape(-1, 1)
    tr = torch.linspace(0, Tmax, n).reshape(-1, 1)
    cond = torch.zeros_like(xl)
    # logger.info(f"sampling in interior, method is {method}")
    return xr.requires_grad_(True), xl.requires_grad_(True), tr.requires_grad_(True), tl.requires_grad_(True), cond

def initial(n=1000):
    xx = torch.linspace(-1., 1., n).reshape(-1, 1)*half_x
    tt = torch.zeros(n).reshape(-1, 1)
    cond = 0.1 / torch.sqrt(alpha * (tt + t_start)) * torch.exp(-(xx+2-4*tt) ** 2 / (4 * alpha * (tt + t_start)))
    # logger.info(f"sampling in interior, method is {method}")
    return xx.requires_grad_(True), tt.requires_grad_(True), cond


loss = torch.nn.MSELoss()

import math


def l_interior(u, method='random', resample=False):
    if resample or ('interior' not in collocations):
        x, t, cond = interior(method=method)
        collocations['interior'] = (x, t, cond)
    x, t, cond = collocations['interior']
    lhs = compute_lhs(u, x, t)
    l = loss(lhs, cond)
    return l


def l_boundary(u):
    if 'boundary' not in collocations:
        xr, xl, tr, tl, cond = boundary()
        collocations['boundary'] = (xr, xl, tr, tl, cond)
    xr, xl, tr, tl, cond = collocations['boundary']
    return loss(u(torch.cat([xr, tr], dim=1)), cond) + loss(u(torch.cat([xl, tl], dim=1)), cond)


def l_initial(u):
    if 'initial' not in collocations:
        x, t, cond = initial()
        collocations['initial'] = (x, t, cond)
    x, t, cond = collocations['initial']
    return loss(u(torch.cat([x, t], dim=1)), cond)


def visualize_error(ax, u):
    xx, tt, u_truth = exact_heat()
    shape = u_truth.shape
    xx = torch.Tensor(xx)
    tt = torch.Tensor(tt)
    u_truth = torch.Tensor(u_truth).reshape(*shape)

    xx = xx.reshape(-1, 1)
    yy = tt.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    with torch.no_grad():
        u_pred = u(xy).reshape(*shape)
        error = torch.abs(u_pred - u_truth)
    xx, tt, _ = exact_heat()
    ax.pcolormesh(xx.reshape(*shape), tt.reshape(*shape), error.detach().cpu().numpy(), vmin=0, vmax=0.03)
    ax.set_title('abs error')


def visualize(ax, u, verbose=True):
    xc = torch.linspace(-1, 1, 100)*half_x
    tc = torch.linspace(0, Tmax, 100)
    xx, yy = torch.meshgrid(xc, tc, indexing='xy')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    u_pred = u(xy)
    if verbose:
        logger.info("L2 error is: {}".format(float(l2_loss(u))))

    u_pred = u_pred.detach().cpu().numpy().reshape(100, 100)

    xx = xx.detach().cpu().numpy().reshape(100, 100)
    yy = yy.detach().cpu().numpy().reshape(100, 100)

    # ax.pcolormesh(xx, yy, np.abs(u_pred-u_truth), cmap='hot', vmin=0, vmax=1)
    ax.pcolormesh(xx, yy, u_pred, vmin=0., vmax=2., cmap='hot')
    ax.set_title('Prediction')


def visualize_scatter(ax, collocation):
    x, y, _ = collocation['interior']
    x = x.detach().cpu().numpy().ravel()
    y = y.detach().cpu().numpy().ravel()
    ax.scatter(x, y, s=1)


def compose_loss(l_interior_val, l_boundary_val, l_init_val):
    return 2 * l_init_val + 2 * l_boundary_val + 0.2 * l_interior_val


def write_res(mse_list):
    with open(path.joinpath('result.csv'), "a+") as f:
        f.write(', '.join(mse_list))
        f.write('\n')


def eval_u(u, mse_list):
    l2_rel = several_error(u)
    mse_list.append(str(l2_rel))
    logger.info(f'mse: {l2_rel}')


def ff():
    """
    FF
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'ff_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("ff")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    collocations['interior'] = interior_ff(np.ones((100, 100)))
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        l_interior_val = l_interior(u)
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        opt.zero_grad()
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'ff.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'ff.pth'))
    write_res(mse_list)
    return mse_list


def ff_resample():
    """
    FF-R
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'ff_resample_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("ff resampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            collocations['interior'] = interior_ff(np.ones((100, 100)))
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        l_interior_val = l_interior(u)
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        opt.zero_grad()
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'ff_re.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'ff_re.pth'))
    write_res(mse_list)
    return mse_list


def ff_rar(mem=0.9):
    """
    RANG-m
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'ff_rar_{mem:.2f}_mse']

    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info(f"ff_rar_{mem:.2f}")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters())
    collocations['interior'] = interior_ff(np.ones((100, 100)))

    for i in range(maxiter):
        opt.zero_grad()

        if i > 0 and i % rar_interval == 0:
            xx, yy, new_error = compute_res(u)
            min_v = np.min(new_error)
            max_v = np.max(new_error)
            new_error = (new_error - min_v) / (max_v - min_v + 1e-8)
            try:
                error = np.maximum(mem * error, new_error)
            except:
                error = new_error
            collocations['interior'] = interior_ff(error, sample_num)

            if verbose:
                logger.info("length of samples is {}".format(len(collocations['interior'][0])))

            if sample_idx % freq_draw == 0:
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        l_interior_val = l_interior(u)
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}, point num is {len(collocations["interior"][0])}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'ff_rar_{mem:.2f}.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'ff_rar_{mem:.2f}.pth'))
    write_res(mse_list)

    return mse_list


def hammersely_sample():
    """
    Hammersley
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'hammersely_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("hammersely sampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        opt.zero_grad()

        l_interior_val = l_interior(u, method='hammersely')
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')

    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'hammersely_evo_ac.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'hammersely_evo_ac.pth'))
    write_res(mse_list)
    return mse_list


def lhs_sample():
    """
    LHS
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'lhs_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("lhs sampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        l_interior_val = l_interior(u, method='lhs')
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        opt.zero_grad()
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'lhs.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'lhs.pth'))
    write_res(mse_list)
    return mse_list


def lhs_resample():
    """
    LHS
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'lhs_resample_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("lhs resampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            l_interior_val = l_interior(u, method='lhs', resample=True)
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        else:
            l_interior_val = l_interior(u, method='lhs')
        opt.zero_grad()
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'lhs_re.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'lhs_re.pth'))
    write_res(mse_list)
    return mse_list


def random():
    """
    Random
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'random_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("random resampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        l_interior_val = l_interior(u, method='random')
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)

        opt.zero_grad()
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'random.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'random.pth'))
    write_res(mse_list)
    return mse_list


def random_resample():
    """
    Random_R
    """
    global exp_id
    global collocations
    mse_list = [str(exp_id), f'random_resample_mse']
    fig2, ax2 = plt.subplots(resample_num // freq_draw, 3, figsize=(12, resample_num // freq_draw * 4))
    fig2.set_tight_layout(True)
    sample_idx = 0
    logger.info("random resampling")
    collocations = dict()
    u = MLP(seq=net_seq)
    opt = torch.optim.Adam(params=u.parameters(), lr=0.001)
    for i in range(maxiter):
        if i > 0 and i % rar_interval == 0:
            l_interior_val = l_interior(u, method='random', resample=True)
            if sample_idx % freq_draw == 0:
                xx, yy, error = compute_res(u)
                visualize(ax2[sample_idx // freq_draw, 0], u, verbose=verbose)
                visualize_scatter(ax2[sample_idx // freq_draw, 1], collocations)
                ax2[sample_idx // freq_draw, 2].pcolormesh(xx, yy, error)
            eval_u(u, mse_list=mse_list)
            sample_idx += 1
        else:
            l_interior_val = l_interior(u, method='random')
        l_boundary_val = l_boundary(u)
        l_initial_val = l_initial(u)
        l = compose_loss(l_interior_val, l_boundary_val, l_initial_val)
        opt.zero_grad()
        l.backward()
        opt.step()
        if i % log_interval == 0:
            logger.info(f'iteration {i}: loss is {float(l)}')
    eval_u(u, mse_list=mse_list)
    fig2.savefig(path.joinpath(f'random_re.png'))
    plt.close(fig2)
    torch.save(u.state_dict(), path.joinpath(f'random_re.pth'))
    write_res(mse_list)
    return mse_list


if __name__ == '__main__':
    exp_id = int(args.start_epoch)
    for i in range(int(args.repeat)):
        ff() # FF
        exp_id += 1

        ff_resample() # FF-R
        exp_id += 1

        ff_rar(0.9) # RANG-m
        exp_id += 1

        ff_rar(0.0) # RANG
        exp_id += 1

        hammersely_sample() # Hammersley
        exp_id += 1

        lhs_sample() # LHS
        exp_id += 1

        lhs_resample() # LHS-R
        exp_id += 1

        random() # Random
        exp_id += 1

        random_resample() # Random-R
        exp_id += 1