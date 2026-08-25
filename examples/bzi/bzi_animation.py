import sys
sys.path.append(".")
import numpy as np
from khepri.crystal import Crystal

import numpy as np
import matplotlib.pyplot as plt
from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from khepri.tools import compute_kplanar
import sys
from khepri.beams import amplitudes_from_fields
from khepri.misc import coords
from matplotlib.animation import FuncAnimation
from khepri.layer import Field


bzx, bzy = 41, 1

NS = 31
wl = 1.1
w0 = 2 * wl
zmax=28
pw = (7, 1)
theta = np.deg2rad(25)
eps1 = 1
eps2 = 4

from khepri.beams import gen_bzi_grid

kbz = gen_bzi_grid((bzx, bzy))
print(kbz.shape)
kbz = kbz.reshape(2, -1).T


kxi, kyi = compute_kplanar(eps1, wl, np.rad2deg(theta), 0)
kbz[:, 0] += kxi.real
kbz[:, 1] += kyi.real

X = np.linspace(0, bzx, NS*bzx, endpoint=True)
Y = np.linspace(0, bzy, NS*bzy, endpoint=True)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)
source_real=shifted_rotated_fields(
    _paraxial_gaussian_field_fn, X, Y, Z,
    wl, bzx/2, bzy/2, -zmax/2, theta, 0, np.pi/2, beam_waist=w0, er=eps1)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)
canvas_size = (512,512)

e1 = Expansion(pw)
cvn = Drawing((128,128), eps1)
cvn.rectangle((0,0), (0.5, 1), eps2)
fields = None
if True:
    for kp in kbz:
        cl = Crystal(pw, epse=eps2, epsi=eps1)
        cl.add_layer_uniform("S1", eps1, 0.99)
        #cl.add_layer_uniform("S3", 4.0, 0.4)
        cl.add_layer_pixmap("S3", cvn.canvas(), 0.4)
        cl.add_layer_uniform("S2", eps2, 16.99)
        stack =["S1"]*14
        stack.extend(["S3"]*1)
        stack.extend(["S2"])
        cl.set_device(stack, [True]*len(stack))
        cl.set_source(wl, np.nan, np.nan, kp=kp)
        cl.solve()
        F = amplitudes_from_fields(source_real, e1, wl, kp, X, Y, (bzx, bzy))
        x, y, z = coords(0, bzx, bzy/2, bzy/2, 0.01, zmax, (3, 512, 512))
        S, U = np.split(F.flatten(), 2)
        F  = S, U #np.zeros_like(U)
        E, H = cl.fields_volume(x, y, z, incident_fields=F)
        if fields is None:
            fields = np.asarray((E,H))
        else:
            fields += np.asarray((E,H))
    np.save("fields.npy", fields)

fields = np.load("fields.npy")
if False:
    fields = np.swapaxes(fields, 3, 2)
    E = fields[0, :, :, :, 1] # (Z, F, X)
    H = fields[1, :, :, :, 1]
    print(E.shape)
    PT = np.cross(E, np.conj(H/1j)) # 1j because of normalization
    x, y, z = coords(0, bzx, bzy/2, bzy/2, 0.01, zmax, (3, 256, 256))
    PTnorm = np.sum(np.power(PT, 2), axis=-1).real
    fig, ax = plt.subplots(figsize=(8,6), dpi=250)
    vmax = np.max(np.abs(PTnorm.real))
    image = ax.matshow(PTnorm, cmap="magma",extent=[0, np.max(x), 0, zmax], vmin=0, vmax=vmax)
    plt.colorbar(image)
    plt.savefig("test.png")
else:
    K = np.exp(1j)
    fields = fields[0, :, 1, :] # Take Ey
    fig, ax = plt.subplots(figsize=(8,5.5), dpi=250)
    vmax = np.max(np.abs(fields.real)) 
    image = ax.matshow(fields.real[:, :, 1], cmap="RdBu",extent=[0, bzx, 0, zmax], vmin=-vmax, vmax=vmax) # We chart zx plane
    plt.savefig("test.pdf")
    def update(frame):
        image.set_data((fields[:, :, 1] * K**frame).real)
        return image,

    epsdisp = cvn.canvas()[64,:]
    epsdisp /= epsdisp.max()
    epsdisp *= 0.8
    epsdisp += 0.99*4 - 0.8
    ax.axhline(0.99*14, color="k")
    ax.arrow(23.5, 22, -5*np.sin(theta), -5*np.cos(theta), head_width=0.5,head_length=0.5,length_includes_head=True,color="green")
    m = np.array([-2, -1, 0, 1, 2, 3])
    anglest = np.arcsin(np.sqrt(eps1)/np.sqrt(eps2)*np.sin(theta) - m * wl/np.sqrt(eps2))
    anglesr = np.arcsin(np.sin(theta) - m * wl/np.sqrt(eps1))
    for angle in anglest:
        ax.arrow(20, 14, -5*np.sin(angle), -5*np.cos(angle), head_width=0.5,head_length=0.5,length_includes_head=True,color="black")
    for angle in anglesr:
        ax.arrow(20, 14, -5*np.sin(angle), 5*np.cos(angle), head_width=0.5,head_length=0.5,length_includes_head=True,color="black")
    ax.set_xlabel("X [um]")
    ax.set_ylabel("Z [um]")
    ax.xaxis.tick_bottom()
    ax.text(0.5, 27, "Corrugated interface")
    ax.text(0.5, 26, "$n_1$ = 1")
    ax.text(0.5, 25, "$n_2$ = 2")
    ax.text(0.5, 24, "$\\theta$ = 25")
    ax.text(0.5, 23, "$\\Lambda=1$um")
    ax.text(0.5, 22, r"$\lambda=1.1\Lambda$")
    ax.text(0.5, 21, "$\\lambda_{dc}=0.5$")
    ax.text(0.5, 20, "$h=0.3$")
    ax.text(0.5, 19, "$\\vec p=\\vec y$")
    ax.text(0.5, 18, "$pw=(7, 1)$")
    ax.text(0.5, 17, "$bz=(41, 1)$")
    ax.text(0.5, 16, "walltime: 5s")
    ax.text(0.5, 15, "$E_y$")
    #ax.axhline(0.99*4+0.8, color="k")
    #plt.plot(np.linspace(0, bzx, 128*bzx), np.tile(epsdisp, bzx), color="k")
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 10, endpoint=False))
    ani.save("anim.gif")
