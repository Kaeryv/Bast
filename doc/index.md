[Reusable scientific validation cases](scientific-validation.md)

<a id="tcell"></a>

# tcell

<a id="read_ex"></a>

# read\_ex

<a id="plot_data"></a>

# plot\_data

Python script example for plotting Tidy3D simulation results using GUI exported data file.
Please modify according to the specific dataset you download.

<a id="khepri"></a>

# khepri

<a id="khepri.ez"></a>

# khepri.ez

<a id="khepri.ez.load_layers"></a>

#### load\_layers

```python
def load_layers(state_dict)
```

Load the layers from the dictionary.
Removes some user-friendly notations/segregations

<a id="khepri.misc"></a>

# khepri.misc

<a id="khepri.crystal"></a>

# khepri.crystal

<a id="khepri.crystal.Crystal"></a>

## Crystal Objects

```python
class Crystal()
```

<a id="khepri.crystal.Crystal.add_layer_uniform"></a>

#### add\_layer\_uniform

```python
def add_layer_uniform(name, epsilon, depth)
```

Add layer without planar structuration. It will be processed analytically
without solving any eigenvalue problem

<a id="khepri.crystal.Crystal.add_layer_pixmap"></a>

#### add\_layer\_pixmap

```python
def add_layer_pixmap(name, epsilon, depth)
```

Add a layer from 2D ndarray that provides eps(x,y). This method will use FFT.

<a id="khepri.crystal.Crystal.add_layer_analytical"></a>

#### add\_layer\_analytical

```python
def add_layer_analytical(name, epsilon, epsilon_host, depth)
```

Add a layer from 2D ndarray that provides eps(x,y). This method will use analytical formulas.

<a id="khepri.crystal.Crystal.set_device"></a>

#### set\_device

```python
def set_device(layers_stack, fields_mask=None)
```

Take the stacking from the user device and pre.a.ppend the incidence and emergence media.

<a id="khepri.crystal.Crystal.locate_layer"></a>

#### locate\_layer

```python
def locate_layer(z)
```

Locates the layer at depth z and also computes the position relative to layer.

**Arguments**:

- `z` _float_ - z depth
  

**Returns**:

- `tuple` - The layer, the index and the relative depth.

<a id="khepri.crystal.Crystal.field_cell_xy"></a>

#### field\_cell\_xy

```python
def field_cell_xy(z, incident_fields, method="fft")
```

Returns the fields in the unit cell for a depth z.

<a id="khepri.crystal.Crystal.get_source_as_field_vectors"></a>

#### get\_source\_as\_field\_vectors

```python
def get_source_as_field_vectors()
```

Construct a Field supervector from the current Crystal source.

**Returns**:

- `tuple` - (E, H) in Fourier space

<a id="khepri.crystal.Crystal.fields_coords_xy"></a>

#### fields\_coords\_xy

```python
def fields_coords_xy(x,
                     y,
                     z,
                     incident_fields=None,
                     use_lu=False,
                     kp=None,
                     return_fourier=False)
```

Returns the fields at specified coordinates (x,y) for a depth z.

**Arguments**:

- `x` _array_ - x coordinates
- `y` _array_ - y coordinates (must be len(x))
- `z` _scalar_ - z depth
- `incident_fields` __type__ - _description_
  

**Returns**:

- `tuple` - contains the E and H fields.

<a id="khepri.fields"></a>

# khepri.fields

<a id="khepri.fields.translate_mode_amplitudes"></a>

#### translate\_mode\_amplitudes

```python
def translate_mode_amplitudes(Sld, c1p, c1m)
```

Sld: scattering matrix left of the layer, including the layer.
c1p: left-side incoming fields on the stack.
c1m: left-side outgoing fields  "   "     "

<a id="khepri.fields.translate_mode_amplitudes2"></a>

#### translate\_mode\_amplitudes2

```python
def translate_mode_amplitudes2(Sl, Sr, c1p, c1m, c2p)
```

Sld: scattering matrix left of the layer, including the layer.
c1p: left-side incoming fields on the stack.
c1m: left-side outgoing fields

<a id="khepri.fields.fourier2real_fft"></a>

#### fourier2real\_fft

```python
def fourier2real_fft(ffield, a, target_resolution=(127, 127), kp=(0, 0))
```

Compute the real-space field from a truncated fourier representation.

<a id="khepri.fields.real2fourier_xy"></a>

#### real2fourier\_xy

```python
def real2fourier_xy(field, kx, ky, x, y)
```

Turns Fourier-space fields into real-space fields using home-made DFT.
This allows to choose when the fields should be evaluated.
This routine is way slower for the whole unit cell.

<a id="khepri.fields.layer_eigenbasis_matrix"></a>

#### layer\_eigenbasis\_matrix

```python
def layer_eigenbasis_matrix(WI, VI)
```

Matrix whose columns are the eigenmodes of the E (upper part) and H  (lower part) fields eigenmodes.
This matrix is used to go back and forth between the eigenmodes coefficients and fields spaces.

<a id="khepri.fields.fourier_fields_from_mode_amplitudes"></a>

#### fourier\_fields\_from\_mode\_amplitudes

```python
def fourier_fields_from_mode_amplitudes(RI,
                                        LI,
                                        R0,
                                        mode_amplitudes,
                                        zbar,
                                        luRI=None)
```

Computes the fields from the mode coefficients at specified z-location (depth).

<a id="khepri.tools"></a>

# khepri.tools

<a id="khepri.tools.unitcellarea"></a>

#### unitcellarea

```python
def unitcellarea(a1, a2)
```

Returns the area given base vectors.

<a id="khepri.tools.reciproc"></a>

#### reciproc

```python
def reciproc(a1, a2)
```

Compute reciproc lattice basis vectors.

<a id="khepri.constants"></a>

# khepri.constants

<a id="khepri.beams"></a>

# khepri.beams

<a id="khepri.factory"></a>

# khepri.factory

<a id="khepri.alternative"></a>

# khepri.alternative

Alternative formulation of RCWA

Work in progress, use with caution.

Notations:
S11 = S[0,0]
S22 = S[1,1]
S = [[S11, S12], [S21, S22]]

<a id="khepri.alternative.solve_uniform_layer"></a>

#### solve\_uniform\_layer

```python
def solve_uniform_layer(Kx, Ky, er, m_r=1)
```

Computes P & Q matrices for homogeneous layer.

<a id="khepri.eigentricks"></a>

# khepri.eigentricks

<a id="khepri.extension"></a>

# khepri.extension

<a id="khepri.extension.joint_subspace"></a>

#### joint\_subspace

```python
def joint_subspace(submatrices: list, kind=0)
```

Wrapper of _joint_subspace that processes 4 quadrants of smatrix

<a id="khepri.layer"></a>

# khepri.layer

<a id="khepri.layer.stack_layers"></a>

#### stack\_layers

```python
def stack_layers(pw, layers, mask)
```

Takes as input a list of layers and a conform list of booleans.
The boolean dictates if we plan on keeping the fields for the layer.

<a id="khepri.layer.Layer"></a>

## Layer Objects

```python
class Layer()
```

<a id="khepri.layer.Layer.pixmap_or_uniform"></a>

#### pixmap\_or\_uniform

```python
@classmethod
def pixmap_or_uniform(cls, expansion, pixmap, depth)
```

This method is a convenience when you don't know what is inside pixmap.
If the structure is rigorously uniform, it will be redirected to uniform solver.
You better filter too small details before sending pixmap to this function.
The expected use case is when doing optimization.

**Arguments**:

- `expansion` _Expansion_ - The expansion to be used for this layer.
- `pixmap` _array_ - A numpy picture of you 2D pattern. Can be uniform.
- `depth` _float_ - The depth of the layer.

<a id="khepri.layer.Layer.pixmap"></a>

#### pixmap

```python
@classmethod
def pixmap(cls, expansion, pixmap, depth)
```

Constructing a layer this way will use the FFT algorithm to source the convolution matrix.
The FFT will be applied on the real-space descritpion of the unit cell dielectric 'pixmap'.

**Arguments**:

- `expansion` _Expansion_ - The expansion to be used for this layer.
- `pixmap` _array_ - A numpy picture of you 2D pattern.
- `depth` _float_ - The depth of the layer.

<a id="khepri.layer.Layer.solve"></a>

#### solve

```python
def solve(k_parallel: Tuple[float, float], wavelength: float)
```

Obtain the eigenspace and S-matrix from layer parameters.
parameters:
k_parallel: incident transverse wavevector.
wavelength: excitation wavelength

<a id="khepri.expansion"></a>

# khepri.expansion

<a id="khepri.expansion.generate_expansion_indices"></a>

#### generate\_expansion\_indices

```python
def generate_expansion_indices(pw)
```

Returns expansion on square grid of reciprocal space.
ndarray.shape = (2, prod(pw))
Note: multiply by reciprocal lattice basis for @hex

<a id="khepri.expansion.Expansion"></a>

## Expansion Objects

```python
class Expansion()
```

<a id="khepri.expansion.Expansion.__add__"></a>

#### \_\_add\_\_

```python
def __add__(rhs)
```

Produces a new expansion that is the Minkowski sum of the two expansions self and rhs.

<a id="khepri.fourier"></a>

# khepri.fourier

<a id="khepri.fourier.idft"></a>

#### idft

```python
@njit(parallel=BAST_MT_ON)
def idft(ffield, kx, ky, x, y)
```

Turns Fourier-space fields into real-space fields using home-made DFT.
This allows to choose when the fields should be evaluated.
This routine is way slower for the whole unit cell.

<a id="khepri.fourier.combine_fourier_masks"></a>

#### combine\_fourier\_masks

```python
def combine_fourier_masks(islands_data, eps_host, inverse=False)
```

Takes as input an array of booleans and their associated epsilon value.
Outputs the resulting fourier transform.

<a id="khepri.draw"></a>

# khepri.draw

A class for designing RCWA's 2D extruded patterns.
This class can be replaced by you numpy drawing skills.
Note that:
    arrays are indexed a[x,y].

<a id="khepri.draw.Drawing"></a>

## Drawing Objects

```python
class Drawing()
```

<a id="khepri.draw.Drawing.parallelogram"></a>

#### parallelogram

```python
def parallelogram(xy, bh, tilt_rad, epsilon)
```

Parallelogram with basis parallel to x axis.

<a id="khepri.draw.Drawing.from_numpy"></a>

#### from\_numpy

```python
def from_numpy(X)
```

Will convert numpy array geometry to islands description.
This has limits and only works for 1D array.

<a id="khepri.gui"></a>

# khepri.gui
