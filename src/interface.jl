# NSEBase method extensions for SquareDuctGrid.
#
# NSEBase treats inhomogeneous directions as abstract: `dd!` and
# `inhomogeneous_laplacian!` throw NotImplementedError for non-FFT dims and
# expect downstream packages to extend them with grid-specific matrix
# applications. These are those extensions.
#
# The duct has two inhomogeneous directions:
#   - x → storage dim 1  (DUCT_INHOMOGENEOUS_DIMS[1])
#   - y → storage dim 2  (DUCT_INHOMOGENEOUS_DIMS[2])
#
# Both use the same operator D₁ (or D₁⁺), so dd! needs only one method.
# FDGrids.mul! routes by the Val(DIM) argument passed from the caller.
#
# `parent(out)` is unwrapped before every `mul!` to get a plain array.
# FTField does not implement Base.strides, so views of FTField are not
# recognised as strided arrays by BLAS; unwrapping first ensures mul! dispatches
# to the FDGrids kernel rather than generic BLAS.

# ------------------------------------------------------------------ #
# First-order derivative                                              #
# ------------------------------------------------------------------ #

"""
    NSEBase.dd!(out, u, ::Val{DIM}; adjoint=false)

Apply the cross-section derivative in storage dimension `DIM` (1 for x, 2 for y).

Both inhomogeneous directions use the same operator `D₁` (or `D₁⁺` when
`adjoint=true`); `FDGrids.mul!` routes the application along the correct
storage dimension via the `Val(DIM)` argument.
"""
function NSEBase.dd!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G},
                     dim::Val{DIM};
                     adjoint::Bool = false) where {DIM, G<:AbstractSquareDuctGrid{<:Any, <:Any, NSEBase.Undecomposed}}
    D = adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁
    LinearAlgebra.mul!(parent(out), D, parent(u), dim)
    return out
end

# ------------------------------------------------------------------ #
# Laplacian                                                           #
# ------------------------------------------------------------------ #

"""
    NSEBase.inhomogeneous_laplacian!(out, u; adjoint=false)

Apply the cross-section (x + y) contribution to the Laplacian of `u`.

Computes `(D₂ applied along dim 1) + (D₂ applied along dim 2)` and writes the
combined result into `out`. The homogeneous spectral contribution (z wavenumber
terms) is added later by `NSEBase.add_homogeneous_laplacian!`.

The two terms are accumulated using `FDGrids.mul!` with `Val(false)` (overwrite)
followed by `Val(true)` (accumulate), which avoids any intermediate allocation.
"""
function NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G};
                                          adjoint::Bool = false) where {G<:AbstractSquareDuctGrid{<:Any, <:Any, NSEBase.Undecomposed}}
    D2 = adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂
    # Overwrite out with D₂ applied along dim 1 (∂²/∂x²).
    LinearAlgebra.mul!(parent(out), D2, parent(u),
                       Val(DUCT_INHOMOGENEOUS_DIMS[1]), Val(false))
    # Accumulate D₂ applied along dim 2 (∂²/∂y²) into out.
    LinearAlgebra.mul!(parent(out), D2, parent(u),
                       Val(DUCT_INHOMOGENEOUS_DIMS[2]), Val(true))
    return out
end
