<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ReSolverSquareDuct.jl logo" width="680">
</p>

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Davide-Lasagna-s-Lab.github.io/ReSolver-SquareDuct.jl/dev/)

# ReSolverSquareDuct.jl

`ReSolverSquareDuct.jl` provides square-duct cross-section grids, wall-normal
derivative hooks, and a convenience equation constructor for pressure-driven
square-duct flow, built on top of `NSEBase.jl`.

The package targets spectral and resolvent workflows with two inhomogeneous
cross-section directions (`x` and `y`, both collocation) and homogeneous
streamwise (`z`, rfft) and temporal (`t`, complex FFT) directions.

## Features

- `SquareDuctGrid` for square-duct storage with physical sizes `(N, N, Nz, Nt)`,
  where both cross-section directions share the same `N`-point collocation grid.
- NSEBase-compatible `points`, `weights`, `growto`, and derivative hooks.
- Single shared set of FD differentiation operators for both cross-section
  directions, constructed automatically from any `FDGrids.AbstractGridDistribution`.
- Allocation-free inhomogeneous Laplacian via the `FDGrids` 5-arg `mul!` accumulate form.
- `SquareDuctFlow` equation constructor for pressure-driven turbulence computations.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ReSolver-SquareDuct.jl")
```

## Example

```julia
using FDGrids
using ReSolverSquareDuct

# 32-point cross-section, width-7 stencil, 63 streamwise modes, Lz = 4π
g = SquareDuctGrid(32, 7, 63, 1, 2π/4π;
                   dist = FDGrids.GaussLobattoGrid())

# Pressure-driven equations at Re = 2000
equations = SquareDuctFlow(g, 2000)
```

The cross-section occupies `x, y ∈ [0, 1]` with Chebyshev-Lobatto nodes (endpoints
included for no-slip boundary conditions). Differentiation matrices and their
quadrature-weighted adjoints are built automatically from the supplied distribution.

## Documentation

The full documentation is available at the
[development documentation site](https://Davide-Lasagna-s-Lab.github.io/ReSolver-SquareDuct.jl/dev/).
