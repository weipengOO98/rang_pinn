import numpy as np
import math
from pyDOE import lhs as _lhs

def scatter_halftone(box, ninit, dotmax, radius):
    lb = box[0]
    rb = box[1]
    db = box[2]
    ub = box[3]
    count=0
    dotnr = -1
    N_PDP_MAX = 100000
    pdp_x = np.zeros(N_PDP_MAX)
    pdp_y = np.zeros(N_PDP_MAX)

    pdp_x[:ninit] = np.linspace(lb, rb, ninit)
    pdp_y[:ninit] = np.random.rand(ninit) * 1e-4 + db

    pdp_num = ninit
    xy = np.zeros((dotmax, 2))
    i = np.argmin(pdp_y[:ninit])
    ym = pdp_y[i]
    fan = np.linspace(0.1, 0.9, 5)
    while ym <= ub and dotnr < dotmax:
        dotnr += 1
        xy[dotnr, 0] = pdp_x[i]
        xy[dotnr, 1] = pdp_y[i]
        r = radius(xy[dotnr, :])
        dist2 = (pdp_x[:pdp_num] - pdp_x[i]) ** 2 + (pdp_y[:pdp_num] - pdp_y[i]) ** 2

        ileft = np.where(dist2[:i] > r ** 2)
        if len(ileft[0]) == 0:
            ileft = -1
            ang_left = np.pi
        else:
            ileft = max(ileft[0])
            ang_left = np.arctan2(pdp_y[ileft] - pdp_y[i], pdp_x[ileft] - pdp_x[i])

        iright = np.where(dist2[i:pdp_num] > r ** 2)
        if len(iright[0]) == 0:
            iright = -1
            ang_right = 0
        else:
            iright = min(iright[0])
            ang_right = np.arctan2(pdp_y[i + iright] - pdp_y[i], pdp_x[i + iright] - pdp_x[i])
        ang = ang_left - fan * (ang_left - ang_right)
        pdp_new_x = pdp_x[i] + r * np.cos(ang)
        pdp_new_y = pdp_y[i] + r * np.sin(ang)
        ind = np.logical_and(pdp_new_x <= rb, pdp_new_x >= lb)
        pdp_new_x = pdp_new_x[ind]
        pdp_new_y = pdp_new_y[ind]
        new_add = len(pdp_new_x)
        if iright ==-1 and ileft == -1:
            removed = pdp_num
        elif iright ==-1:
            removed = pdp_num-ileft-1
        elif ileft == -1:
            removed = iright-1+i-ileft
        else:
            removed = i-ileft+iright-1
        if iright!=-1:
            pdp_x[iright + i + new_add - removed:pdp_num + new_add - removed] = pdp_x[iright + i:pdp_num]
            pdp_y[iright + i + new_add - removed:pdp_num + new_add - removed] = pdp_y[iright + i:pdp_num]

        pdp_x[ileft + 1:ileft + 1 + new_add] = pdp_new_x
        pdp_y[ileft + 1:ileft + 1 + new_add] = pdp_new_y

        pdp_num = pdp_num + new_add - removed
        i = np.argmin(pdp_y[:pdp_num])
        ym = pdp_y[i]

    xy = xy[:dotnr, :]
    return xy

def error_ff(target_num, error, max_min_density_ratio=20, box=None, sdf=None):
    _N = len(error)
    if box is None:
        box = [0, 1, 0, 1]
    if sdf is None:
        sdf = lambda x: np.ones((len(x), 1))
    error = -error
    error_min = np.min(error)
    error_max = np.max(error)

    error = ((error - error_min) / ((error_max - error_min) + 1e-10))

    min_scale = 0.02
    max_scale = 1.
    scale = (min_scale + max_scale) / 2

    def r(xy):
        ixy = np.asarray(np.round(xy * (_N - 1)), dtype=int)
        return (error[ixy[1], ixy[0]] * (1 - 1 / math.sqrt(max_min_density_ratio)) + 1 / math.sqrt(
            max_min_density_ratio)) * scale

    xy = scatter_halftone(box, 100, 10000, r)
    len_xy = len(xy)
    while np.abs(len_xy - target_num) / target_num > 0.05 and max_scale - min_scale > 0.003:
        if target_num > len_xy:
            max_scale = scale
        else:
            min_scale = scale
        scale = (max_scale + min_scale) / 2

        def r(xy):
            ixy = np.asarray(np.round(xy * (_N - 1)), dtype=int)
            return (error[ixy[1], ixy[0]] * (1 - 1 / math.sqrt(max_min_density_ratio)) + 1 / math.sqrt(
                max_min_density_ratio)) * scale

        xy = scatter_halftone([0, 1, 0, 1], 100, 10000, r)
        xy = xy[sdf(xy).ravel() > 0, :]
        len_xy = len(xy)
    return xy


def halton(b):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d


def hammersely(Nsize, p=2):
    y = []
    for i, num in enumerate(halton(p)):
        if i >= Nsize:
            break
        y.append(num)
    x = np.arange(0, Nsize) / Nsize
    return np.array([x, y]).T

def lhs(Nsize):
    xy = _lhs(2, Nsize)
    return xy