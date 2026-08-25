import sys
sys.path.append(".")
import numpy as np
from khepri.crystal import Crystal
import matplotlib.pyplot as plt
from khepri.draw import Drawing
from khepri.layer import Layer
from khepri.expansion import Expansion
from khepri.extension import ExtendedLayer as EL
from khepri.beams import _paraxial_gaussian_field_fn, shifted_rotated_fields, _paraxial_laguerre_gaussian_field_fn
from khepri.tools import compute_kplanar
import sys
from tqdm import tqdm
from khepri.beams import gen_bzi_grid
from khepri.tools import rotation_matrix as rot
from khepri.tools import reciproc
from khepri.misc import coords
import psutil
import os

import time

script_start_time = time.time_ns()

process = psutil.Process(os.getpid())
fdepth = float(sys.argv[6])
z0=0
w0=3
offsetx = 30
offsety = 30

twist_angle=np.deg2rad(float(sys.argv[3]))


def size(angle_deg):
    angle = np.deg2rad(angle_deg)
    return float(sys.argv[5])/((1/np.cos(angle))-1)

def moire_lattice(angle):
    AM = rot(angle/2)@ np.eye(2)
    AM *= size(np.rad2deg(angle))
    return AM

twisted = twist_angle != 0
cell_size =  size(twist_angle) if twisted else 1
print(cell_size)

bzi = int(sys.argv[2])
pwx = int(sys.argv[3])
pw = (pwx, pwx)

kbz = gen_bzi_grid((bzi, bzi), a=cell_size, reciproc=None).reshape(2, -1).T

e1 =Expansion(pw)
e2 =Expansion(pw)
e2.rotate(twist_angle)
etw = e1+e2
e = etw
gnorm =  np.linalg.norm(e.g_vectors, axis=0)
asrt = np.argsort(gnorm)
kmin = gnorm[asrt[2]]
cell_size = 4*np.pi / kmin
print(cell_size)
kbz = gen_bzi_grid((bzi, bzi), a=cell_size, reciproc=None).reshape(2, -1).T

if False:
    fig, ax = plt.subplots()
    t = np.linspace(0,2*np.pi, 100)
    e.plot(ax=ax)
    plt.plot(kmin/2*np.cos(t), kmin/2 * np.sin(t), 'k-')
    ax.scatter(*kbz.T, c='b')
    plt.show()

cell_size *= float(sys.argv[5])

NS = 41
wl = float(eval(sys.argv[7])) 
theta = 0

'''
    Build and solve the crystal
'''
from khepri.beams import amplitudes_from_fields
from khepri.alternative import incident

fields = list()
pattern = Drawing((128,128), 4)
pattern.disc((0,0), 0.25, 1)


'''
    Sources computation
'''
x = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
y = np.linspace(0, cell_size*bzi, NS*bzi, endpoint=True)
X, Y = np.meshgrid(x,y, indexing="ij")
Z = np.zeros_like(X)
source_real=shifted_rotated_fields(_paraxial_gaussian_field_fn, X, Y, Z, wl, 
        np.max(x)/2+offsetx, np.max(y)/2+offsety, z0, theta, 0.0, 0.5*np.pi/2, beam_waist=w0)
#source_real = shifted_rotated_fields(_paraxial_laguerre_gaussian_field_fn, X, Y, Z, wl, 
#        np.max(x)/2, np.max(y)/2, z0, theta, 0, np.pi/4, l=0, p=0, w0=w0)

source_real = np.asarray(source_real)
source_real = np.swapaxes(source_real, 0, 2)
source_real = np.swapaxes(source_real, 1, 3)
source_real[..., :, 1] = 1j * source_real[..., :, 0]
print(source_real.shape)


TMPFILE="tw_fields.npz"
if not os.path.isfile(TMPFILE):
    for kp in tqdm(kbz):
        cl = Crystal.from_expansion(etw)
        ss = 1
        #cl.a = cell_size
        cl.add_layer("Sref",   Layer.half_infinite(etw, "reflexion", 1),    False)
        cl.add_layer("S1",     Layer.pixmap(e1, pattern.canvas(), 0.2/ss),  True  )
        cl.add_layer("Si",     Layer.uniform(e1, 1.0,  0.3/ss),              True )
        cl.add_layer("Si2",    Layer.uniform(etw, 1.0,  4),               False)
        cl.add_layer("S2",     Layer.pixmap(e2, pattern.canvas(), 0.2/ss),   True )
        cl.add_layer("Strans", Layer.half_infinite(etw, "transmission", 1), False)
        stack = []
        stack.extend(["S1"]*ss)
        stack.extend(["Si"]*ss)
        stack.extend(["S2"]*ss)
        stack.extend(["Si2"]*2)
        cl.set_device(stack, [True]*len(stack))
        cl.set_source(wl, np.nan, np.nan, kp=kp)
        cl.solve()
        expansion = etw
    
        F = amplitudes_from_fields(source_real, expansion, wl, kp, X, Y, (bzi, bzi), a=cell_size)
    
        #F = np.split(F.flatten(), 2)[0] 
        F = F.flatten()
        if sys.argv[1] == "lg_field":
            y0 = 0
            co = 1
            x, y, z = coords(cell_size*co, cell_size*(bzi-co), y0, y0, 0.001, 3, (512, 1, 256))
            E, H = cl.fields_volume(y, x, z, incident_fields=F) # Progress bar on z axis
        
        elif sys.argv[1] == "tr_field":
            co = 2
            x = np.linspace(cell_size*co, cell_size*(bzi-co), 256)
            y = np.linspace(cell_size*co, cell_size*(bzi-co), 256)
            x, y = np.meshgrid(x, y, indexing="ij")
            E, H = cl.fields_coords_xy(x, y, fdepth, F)
        
        fields.append((E, H))

    fields = np.asarray(fields)
    np.savez_compressed(TMPFILE, fields=fields, x=x, y=y, z=z)

else:
    db = np.load(TMPFILE)
    fields = db["fields"]
    x,y,z = (db[e] for e in ["x", "y", "z"])

if False:
    fig, axs = plt.subplots(bzi, bzi)
    fig2, axs2 = plt.subplots(bzi, bzi)
    for field, ax1, ax2, kp in zip(fields, axs.flat,axs2.flat, kbz):
        f = field[0, 0, :, :]
        nf = np.abs(f)
        nf /= np.max(nf)
        ax1.matshow(np.angle(f), cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax2.matshow(np.abs(f), cmap="magma")
        ax2.axis("off")
        ax1.axis("off")
    fig2.savefig("explode_bzi.png")
    fig.savefig("explode_bzi_angle.png")
    plt.show()


fields = fields.sum(0)


if sys.argv[1] == "lg_field":
    fig, axs = plt.subplots(2, figsize=(5,5))
    ex = fields[0, :, 0, 0, :]
    ey = fields[0, :, 1, 0, :]
    fields=ex
    
    normfields = np.abs(fields)
    normfields /= np.max(normfields)
    extent=[np.min(x), np.max(x), np.min(z), np.max(z)]
    image = axs[0].matshow(np.abs(fields), cmap="magma",extent=extent, aspect=5, origin="lower")
    plt.colorbar(image)
    image = axs[1].matshow(np.angle(fields), cmap="hsv",extent=extent, aspect=5, origin="lower",alpha=normfields, vmin=-np.pi, vmax=np.pi) #, vmin=-1, vmax=1
    plt.colorbar(image)
    for e in [0.00001, 0.2, 0.5, 0.7]:
        axs[0].axhline(e, color="w", lw=1)

else:
    fig, axs = plt.subplots(2, 2, figsize=(5,5))
    ex = fields[0, 0, :, :]
    ey = fields[0, 1, :, :]

    ercp = ex - 1j*ey
    elcp = ex + 1j*ey
    fields=ex
    
    normfields = np.abs(fields)
    normfields /= np.max(normfields)
    extent=[np.min(x), np.max(x), np.min(y), np.max(y)]
    image = axs[0,0].matshow(np.abs(fields), cmap="magma",extent=extent, origin="lower")
    plt.colorbar(image)
    image = axs[0,1].matshow(np.angle(fields), cmap="hsv",extent=extent, alpha=normfields, origin="lower", vmin=-np.pi, vmax=np.pi) #, vmin=-1, vmax=1
    plt.colorbar(image)
    
    sr = source_real[..., 0, 0]
    normsr = np.abs(sr)
    normsr /= np.max(normsr)
    image = axs[1,0].matshow(normsr, cmap="hot",extent=extent)
    image = axs[1,1].matshow(np.angle(sr), cmap="hsv",extent=extent, alpha=normsr)
    axs[0,0].set_xlabel("Length [µm]")
    axs[0,0].set_ylabel("Width [µm]")

fig.savefig("vortex_chiral.png")

script_end_time = time.time_ns()
print((script_end_time - script_start_time) * 1e-9, "s elapsed.")
