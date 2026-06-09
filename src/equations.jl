# Body forces and flow-level equation constructors for square-duct flow.
# CartesianPrimitive3DNSE / CartesianPrimitive3DLNSE live in NSEBase.

# ------------------------------------------------------------------ #
# Constant pressure-gradient forcing                                  #
# ------------------------------------------------------------------ #

"""
    ConstantForcing(value=1)

Body force representing a spatially uniform mean streamwise pressure gradient.

Applied to the zero-wavenumber (mean-flow) mode of the third (z, streamwise)
velocity component at every `(x, y)` collocation point. For the square duct the
mean mode is `WaveNumberVector(0, 0)` because there are two FFT dimensions: z
(rfft, dim 3) and t (complex FFT, dim 4).

`value` is the forcing amplitude. For a duct non-dimensionalised with the
friction velocity `u_τ` and duct half-width `h` (`Re = u_τ h / ν`), the
conventional value is `1` (the default).
"""
struct ConstantForcing{T}
    value :: T
end
ConstantForcing() = ConstantForcing(1.0)

function (f::ConstantForcing)(out::NSEBase.VectorField{N, <:NSEBase.FTField{<:AbstractSquareDuctGrid}},
                              _,
                              ::NSEBase.Mode) where {N}
    # out[3] is the z-velocity (streamwise, physical dim 3).
    # Two FFT dims (z and t) → WaveNumberVector needs two entries.
    mean_mode = out[3][NSEBase.WaveNumberVector(0, 0)]
    mean_mode .+= f.value
    return out
end

# ------------------------------------------------------------------ #
# Flow constructor                                                     #
# ------------------------------------------------------------------ #

"""
    SquareDuctFlow(g, Re; base=(nothing, nothing, nothing), f=1,
                   mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                   dealias=true) -> ProjectedNSE

Construct a `ProjectedNSE` for pressure-driven square-duct flow at Reynolds
number `Re` on the duct grid `g`.

The linearised operator defaults to `NSEBase.AdjointDiscrete`, which is
necessary for exact resolvent and direct-adjoint-looping computations.

# Keyword arguments

- `base`: laminar base flow as a 3-tuple `(U, V, W)`, one entry per velocity
  component (x, y, z). Use `nothing` for absent components. Defaults to no base
  flow (`(nothing, nothing, nothing)`), which means the nonlinear operator acts
  on the full velocity. Pass the laminar Poiseuille solution as `W` to activate
  linearised computations. `W` should be an `N × N` matrix (or a callable
  `(x, y) -> value`) evaluated at the grid's collocation points.
- `f`: amplitude of the mean pressure-gradient [`ConstantForcing`](@ref).
  Defaults to `1`, the conventional value when `Re = Re_τ`.
- `mode`: adjoint mode for the linearised operator.
  Defaults to `NSEBase.AdjointDiscrete()`.
- `fftw_flags`: FFTW planner flags. Defaults to `FFTW.EXHAUSTIVE`.
- `dealias`: allocate physical-space caches on the 3/2-dealiased grid.
  Defaults to `true`.

# Example

```julia
using FDGrids
g  = SquareDuctGrid(32, 5, 63, 1, 2π/4π)
eq = SquareDuctFlow(g, 1000)
```
"""
function SquareDuctFlow(g  :: AbstractSquareDuctGrid,
                        Re :: Real;
                        base :: Tuple   = (nothing, nothing, nothing),
                        f    :: Real    = 1,
                        mode            = NSEBase.AdjointDiscrete(),
                        fftw_flags      = FFTW.EXHAUSTIVE,
                        dealias :: Bool = true)
    T     = eltype(g)
    force = ConstantForcing(T(f))
    return NSEBase.construct_equations(g, Re, base, NSEBase.CartesianPrimitive3D();
                                       force   = force,
                                       mode    = mode,
                                       flags   = fftw_flags,
                                       dealias = dealias)
end
