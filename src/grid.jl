# Square-duct grid with two inhomogeneous cross-section directions.
#
# The duct cross-section occupies x ∈ [0, 1] × y ∈ [0, 1] with no-slip walls
# on all four sides. Both cross-section directions share the same 1D FDGrids
# distribution (Nx = Ny = N, identical collocation points and differentiation
# matrices). The streamwise direction z and temporal direction t are homogeneous
# and FFT-transformed.

# ------------------------------------------------------------------ #
# Axis-layout constants                                               #
# ------------------------------------------------------------------ #

"""
`DUCT_AXES`, `DUCT_FFT_ORDER`, and `DUCT_INHOMOGENEOUS_DIMS` define the storage
layout for square-duct grids:

    DUCT_AXES               = (1, 2, 3, 4)   # (x, y, z, t) → storage dims (identity)
    DUCT_FFT_ORDER          = (3, 4)          # rfft: z (dim 3); complex FFT: t (dim 4)
    DUCT_INHOMOGENEOUS_DIMS = (1, 2)          # x → dim 1, y → dim 2 (both collocation)

The 4D storage array has shape `(N, N, Nz, Nt)`:

  - dim 1 (x): cross-section wall-normal, collocation points on [0, 1].
  - dim 2 (y): cross-section wall-normal, collocation points on [0, 1] (same as x).
  - dim 3 (z): streamwise, rfft (non-negative wavenumbers only).
  - dim 4 (t): temporal, full complex FFT.

Physical and storage orderings coincide, so `DUCT_AXES` is the identity permutation.
The two inhomogeneous directions are in the leading dimensions, which keeps the
slab slices contiguous for MPI decomposition along dim 1.
"""
const DUCT_AXES                = (1, 2, 3, 4)
const DUCT_FFT_ORDER           = (3, 4)
const DUCT_INHOMOGENEOUS_DIMS  = (1, 2)

# ------------------------------------------------------------------ #
# Abstract type                                                       #
# ------------------------------------------------------------------ #

"""
    AbstractSquareDuctGrid{S, T, DECOMPOSITION} <:
        NSEBase.AbstractGrid{T, 4, DUCT_AXES, DUCT_FFT_ORDER, DECOMPOSITION}

Abstract supertype for all square-duct grids using the layout defined by
`DUCT_AXES` and `DUCT_FFT_ORDER`.

`S = (N, N, Nz, Nt)` is the pre-dealiasing size in physical-coordinate order
(physical order equals storage order since `DUCT_AXES` is the identity). `T` is
the scalar real type. Both cross-section directions have the same collocation
count `N`.
"""
abstract type AbstractSquareDuctGrid{S, T, DECOMPOSITION<:NSEBase.GridDecomposition} <:
    NSEBase.AbstractGrid{T, 4, DUCT_AXES, DUCT_FFT_ORDER, DECOMPOSITION} end

Base.size(g::AbstractSquareDuctGrid{S}) where {S} = NSEBase.to_storage_order(S, g)

# ------------------------------------------------------------------ #
# Concrete grid                                                       #
# ------------------------------------------------------------------ #

"""
    SquareDuctGrid{S, T}

Concrete square-duct grid. Both cross-section directions use the same 1D
FDGrids distribution, so a single set of differentiation operators is shared
between `x` (storage dim 1) and `y` (storage dim 2).

`S = (N, N, Nz, Nt)` in physical-coordinate order. `T` is the scalar real type.
The storage array has shape `(N, N, Nz, Nt)`.

# Fields

- `xs`: `N` collocation points shared by both cross-section directions (typically
  Chebyshev-Lobatto nodes on [0, 1]).
- `ws`: `N` 1D quadrature weights (stored for MPI slicing; the NSEBase inner
  product uses the 2D outer-product matrix `ws2d`).
- `ws2d`: `N × N` weight matrix `ws * ws'`, returned by `NSEBase.weights`.
- `D₁`, `D₂`: `N × N` first- and second-order FD operators. Applied along
  storage dim 1 for ∂/∂x and along storage dim 2 for ∂/∂y; `FDGrids.mul!`
  routes by the `Val(DIM)` argument.
- `D₁⁺`, `D₂⁺`: their quadrature-weighted L2 adjoints.
- `α`: streamwise wavenumber scale `2π/Lz`.
"""
struct SquareDuctGrid{S, T, XS, D, DA, W} <:
       AbstractSquareDuctGrid{S, T, NSEBase.Undecomposed}
    xs   :: XS    # N collocation points (x and y share this)
    ws   :: XS    # N 1D quadrature weights
    ws2d :: W     # precomputed (N×N) outer-product weight matrix
    D₁   :: D     # first-order FD operator
    D₂   :: D     # second-order FD operator
    D₁⁺  :: DA    # adjoint first-order
    D₂⁺  :: DA    # adjoint second-order
    α    :: T

    function SquareDuctGrid{S, T}(xs   :: XS,
                                  ws   :: XS,
                                  ws2d :: W,
                                  D₁   :: D,
                                  D₂   :: D,
                                  D₁⁺  :: DA,
                                  D₂⁺  :: DA,
                                  α    :: T) where {S, T,
                                                    XS <: AbstractVector{T},
                                                    D  <: AbstractMatrix{T},
                                                    DA <: AbstractMatrix{T},
                                                    W  <: AbstractMatrix{T}}
        N, _, Nz, Nt = S
        (isodd(Nz) && isodd(Nt)) ||
            throw(ArgumentError("Nz and Nt must be odd (dealiased size will be even)"))
        length(xs)    == N      || throw(ArgumentError("xs has wrong length (expected $N)"))
        length(ws)    == N      || throw(ArgumentError("ws has wrong length (expected $N)"))
        size(ws2d)    == (N, N) || throw(ArgumentError("ws2d must be $N×$N"))
        size(D₁)      == (N, N) || throw(ArgumentError("D₁ must be $N×$N"))
        size(D₂)      == (N, N) || throw(ArgumentError("D₂ must be $N×$N"))
        size(D₁⁺)     == (N, N) || throw(ArgumentError("D₁⁺ must be $N×$N"))
        size(D₂⁺)     == (N, N) || throw(ArgumentError("D₂⁺ must be $N×$N"))
        return new{S, T, XS, D, DA, W}(xs, ws, ws2d, D₁, D₂, D₁⁺, D₂⁺, α)
    end
end

# ------------------------------------------------------------------ #
# Constructor                                                         #
# ------------------------------------------------------------------ #

"""
    SquareDuctGrid(N, width, Nz, Nt, α;
                   dist=FDGrids.GaussLobattoGrid(), T=Float64) -> SquareDuctGrid

Construct a `SquareDuctGrid` from a 1D FDGrids distribution.

Both cross-section directions share the same `N`-point grid on [0, 1].
Differentiation matrices are built once and applied along either dim 1 (x) or
dim 2 (y) depending on which derivative is computed.

# Arguments

- `N`: number of collocation points per cross-section direction (Nx = Ny = N).
- `width`: FD stencil width (must be odd, `≥ 3`). Both first- and second-order
  operators use the same stencil width; wider stencils are higher-order but
  require more halo rows for MPI communication.
- `Nz, Nt`: pre-dealiasing point counts in `z` and `t`; both must be odd.
- `α`: streamwise wavenumber scale `2π/Lz`.
- `dist`: `FDGrids.AbstractGridDistribution` selecting the node placement and
  quadrature rule. Defaults to `GaussLobattoGrid()` (Chebyshev-Lobatto nodes
  with Clenshaw-Curtis weights, all positive, endpoints included for no-slip BCs).
- `T`: scalar real type. Defaults to `Float64`.

# Example

```julia
using FDGrids
g = SquareDuctGrid(32, 5, 63, 1, 2π/4π;
                   dist=FDGrids.GaussLobattoGrid())
```
"""
function SquareDuctGrid(N    :: Int,
                        width:: Int,
                        Nz   :: Int,
                        Nt   :: Int,
                        α    :: Real;
                        dist :: FDGrids.AbstractGridDistribution = FDGrids.GaussLobattoGrid(),
                        T    :: Type{<:Real} = Float64)
    xs_ws = FDGrids.grid(N, 0.0, 1.0, dist)
    xs    = Vector{T}(xs_ws.xs)
    ws    = Vector{T}(xs_ws.ws)
    D₁    = FDGrids.DiffMatrix(xs, width, 1; eltype=T)
    D₂    = FDGrids.DiffMatrix(xs, width, 2; eltype=T)
    D₁⁺   = LinearAlgebra.adjoint(D₁, ws)
    D₂⁺   = LinearAlgebra.adjoint(D₂, ws)
    ws2d  = ws * ws'
    return SquareDuctGrid{(N, N, Nz, Nt), T}(xs, ws, ws2d, D₁, D₂, D₁⁺, D₂⁺, T(α))
end

# ------------------------------------------------------------------ #
# convert                                                             #
# ------------------------------------------------------------------ #

_convert_operator(::Type{T}, D::AbstractMatrix{T}) where {T} = D
_convert_operator(::Type{T}, D::AbstractMatrix) where {T} = T.(D)

"""
    convert(::Type{T}, g::SquareDuctGrid) -> SquareDuctGrid

Return a copy of `g` with all arrays converted to scalar type `T`. Returns `g`
unchanged if it already has eltype `T`.
"""
Base.convert(::Type{T}, g::SquareDuctGrid{S, T}) where {S, T} = g
function Base.convert(::Type{T}, g::SquareDuctGrid{S}) where {T, S}
    xs   = Vector{T}(g.xs)
    ws   = Vector{T}(g.ws)
    ws2d = ws * ws'
    return SquareDuctGrid{S, T}(xs, ws, ws2d,
                                _convert_operator(T, g.D₁),
                                _convert_operator(T, g.D₂),
                                _convert_operator(T, g.D₁⁺),
                                _convert_operator(T, g.D₂⁺),
                                T(g.α))
end

# ------------------------------------------------------------------ #
# NSEBase grid interface                                              #
# ------------------------------------------------------------------ #

"""
    points(g::SquareDuctGrid{S}; dealias=false) -> NTuple{4, AbstractArray}

Return broadcastable coordinate arrays in storage order `(x, y, z, t)` (equals
physical order since `DUCT_AXES` is the identity).

When `dealias=false` (default) the homogeneous directions `z` and `t` use the
pre-dealiasing resolutions `S[3] = Nz` and `S[4] = Nt`. When `dealias=true`
those sizes are replaced by the 3/2-dealiased padded sizes.
"""
NSEBase.points(g::SquareDuctGrid{S}; dealias::Bool=false) where {S} =
    if dealias
        padded = NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g))
        NSEBase.points(g, (padded[DUCT_AXES[3]], padded[DUCT_AXES[4]]))
    else
        NSEBase.points(g, (S[3], S[4]))
    end

"""
    points(g::SquareDuctGrid, (Nz, Nt)) -> NTuple{4, AbstractArray}

Return broadcastable coordinate arrays with streamwise resolution `Nz` and
temporal resolution `Nt`.

Shapes in the `(N, N, Nz, Nt)` storage array:

- `x` → `(N, 1, 1, 1)` (dim 1), collocation points `g.xs` on [0, 1].
- `y` → `(1, N, 1, 1)` (dim 2), collocation points `g.xs` on [0, 1].
- `z` → `(1, 1, Nz, 1)` (dim 3), equally spaced on `[0, 2π/α)`.
- `t` → `(1, 1, 1, Nt)` (dim 4), equally spaced on `[0, 1)`.
"""
function NSEBase.points(g::SquareDuctGrid, (Nz, Nt)::NTuple{2, Int})
    _shape(dim, len) = ntuple(d -> d == dim ? len : 1, 4)
    N  = length(g.xs)
    XX = reshape(g.xs,                      _shape(1, N))
    YY = reshape(g.xs,                      _shape(2, N))
    ZZ = reshape(_equal_points(Nz, 2π/g.α), _shape(3, Nz))
    TT = reshape(_equal_points(Nt, 1.0),    _shape(4, Nt))
    # to_storage_order permutes (x, y, z, t) to storage order; for DUCT_AXES=(1,2,3,4)
    # this is a no-op, but kept for consistency with the NSEBase interface contract.
    return NSEBase.to_storage_order((XX, YY, ZZ, TT), g)
end

_equal_points(N, L) = (0:(N - 1)) ./ N .* L

"""
    wavenumber_scale(g::AbstractSquareDuctGrid, dim::Int) -> Real

Return the wavenumber scale for storage dimension `dim`:

- `dim = 3` (z, rfft): returns `g.α = 2π/Lz`.
- `dim = 4` (t, complex FFT): returns `2π` (unit period [0, 1)).
- `dim = 1` or `2` (x or y, inhomogeneous): returns `1` (unused in practice).
"""
@inline function NSEBase.wavenumber_scale(g::AbstractSquareDuctGrid{S, T}, dim::Int) where {S, T}
    dim == DUCT_AXES[3] && return g.α
    dim == DUCT_AXES[4] && return T(2π)
    return one(T)
end

"""
    weights(g::AbstractSquareDuctGrid) -> AbstractMatrix

Return the precomputed `(N, N)` quadrature weight matrix `ws2d = ws * ws'`.

The entry `ws2d[ix, iy]` equals `ws[ix] * ws[iy]` — the weight for the
collocation point `(xs[ix], xs[iy])` in the 2D cross-section inner product.
"""
NSEBase.weights(g::AbstractSquareDuctGrid) = g.ws2d

"""
    growto(g::SquareDuctGrid{S}, (Nz, Nt)) -> SquareDuctGrid

Return a new grid with streamwise and temporal resolutions `(Nz, Nt)`, keeping
the cross-section operators, collocation points, weights, and `α` unchanged.
"""
NSEBase.growto(g::SquareDuctGrid{S, T}, (Nz, Nt)::NTuple{2, Int}) where {S, T} =
    SquareDuctGrid{(S[1], S[2], Nz, Nt), T}(g.xs, g.ws, g.ws2d,
                                             g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.α)

# NSEBase does not define eltype on grids; provide it here so downstream code
# can infer the scalar type from a grid instance without reaching into type params.
Base.eltype(::AbstractSquareDuctGrid{<:Any, T}) where {T} = T
