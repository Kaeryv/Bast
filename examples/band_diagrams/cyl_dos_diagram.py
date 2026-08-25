import sys
sys.path.append('..')
sys.path.append('.')

from argparse import ArgumentParser
parser = ArgumentParser("Computes band diagram for Bragg Mirror like in Molding The Flow Of Light")
parser.add_argument("-pw", nargs="+", default=(2, 2), type=int)
parser.add_argument("-resolution", nargs="+", default=(100, 50), type=int)
parser.add_argument("-eps1", type=float, default=8.9)
parser.add_argument("-eps2", type=float, default=1)
parser.add_argument("-d", type=float, default=0.5)
parser.add_argument("-a", type=float, default=100)
parser.add_argument("-graph", action="store_true")
parser.add_argument("-compute", action="store_true")
parser.add_argument("-name", default="dos_diagram")
args = parser.parse_args()

import pickle

from khepri.tools import nanometers
from khepri.tmat.matrices import multS
from khepri.tmat.lattice import CartesianLattice
from khepri.tmat.scattering import scattering_matrix
from khepri.eigentricks import scattering_splitlr, scattering_det
from tqdm import tqdm
import numpy as np
from itertools import product
import scipy

pw = args.pw
assert len(pw) == 2
a  = nanometers(args.a)
lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, dtype=np.float32)
N = args.resolution[0]
M = args.resolution[1]
wavelengths = 1 / np.linspace(1./nanometers(5000), 1./nanometers(100), N)
# Compute S-matrices
if args.compute:
    Ss = []
    path = lattice.bz_path(['G', 'X', 'M', 'G'], [M, M//2, M], a)
    for wl, kp in tqdm(list(product(wavelengths, path))):
        S, bf = scattering_matrix(pw, lattice, "disc", [0.5*a, 0.5*a, 0.2*a], args.eps1, args.eps2, wl, kp, args.d*a, 3)
        Sl, Sr = scattering_splitlr(S)
        det = scipy.linalg.det(Sl - np.exp(1j*0)  * Sr)
        Ss.append((det, scipy.linalg.det(S), wl, kp))
    
        
    import pickle
    with open(f"tmp/{args.name}.tmp", "wb") as f:
        pickle.dump(Ss, f)

if args.graph:
    import matplotlib.pyplot as plt
    import pickle
    from scipy.signal import argrelextrema
    with open(f"tmp/{args.name}.tmp", "rb") as f:
        Ss = pickle.load(f)
    
    dos = np.asarray([ detP for detP, detS, wl, ky in tqdm(Ss) ]).reshape(N, 3, M)
    np.save(f"./tmp/dos_{args.name}.npy", dos)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))
    wmax = a / np.min(wavelengths)
    wmin = a / np.max(wavelengths)
    extent=[[0, 1, 0,wmax], [1, 1.5, 0, wmax], [1.5, 2.5, 0, wmax]]
    for i in range(3):
        dd = np.flipud(np.abs(dos[:, i, :]))
        dd = np.log10(dd)
        ax.matshow(dd, extent=extent[i], vmin=-5, vmax=2, alpha=0.1)
        for j in range(M):
            pos = argrelextrema(dd[:, j], np.less)
            print(pos)
            width = extent[i][1] - extent[i][0]
            plt.plot(extent[i][0] + width * j * np.ones_like(pos[0]) / M, wmax -  pos[0] / N, 'k.')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, 2.5)
    ax.set_yticks(np.linspace(0.0, wmax, 9, endpoint=True))
    ax.xaxis.tick_bottom()
    ax.set_xticks([0.0, 1.0, 1.5, 2.5])
    ax.set_xticklabels([r"$\Gamma$", "$X$", "M", r"$\Gamma$"])
    ax.set_ylabel(r"$\frac{\omega a}{2\pi c}$", fontsize=12)
    
    _ = [ax.axvline(x) for x in [0.0, 1.0, 1.5, 2.5]]
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
