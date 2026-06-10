# NSEBase and NSEBaseMPIExt method extensions for AbstractSquareDuctGrid.
#
# NSEBase.dd! and NSEBase.inhomogeneous_laplacian! are the hooks NSEBase calls
# for inhomogeneous (non-FFT) spatial directions. NSEBaseMPIExt.derivative_matrix
# is the hook NSEBaseMPIExt calls to obtain the local FD matrix for each MPI rank.
#
# parent() is unwrapped before every mul! call. FTField does not implement
# Base.strides, so views of FTField are not recognised as strided arrays by BLAS;
# unwrapping first ensures mul! dispatches to the FDGrids kernel.

"""
    NSEBase.dd!(out, u, Val(dim); adjoint=false)

Apply a first-order finite-difference derivative in storage dimension `dim`.

`dim == 1` is the duct `x` direction and `dim == 2` is the `y` direction.
Both directions share the same operator `D₁` (or `D₁⁺` when `adjoint=true`);
`FDGrids.mul!` routes the matrix application along the correct dimension via
the `Val(dim)` argument.
"""
function NSEBase.dd!(out::NSEBase.FTField{G},
                     u::NSEBase.FTField{G},
                     dim::Val{DIM};
                     adjoint::Bool=false) where {DIM, G<:AbstractSquareDuctGrid{<:Any, <:Any, NSEBase.Undecomposed}}
    D = adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁
    LinearAlgebra.mul!(parent(out), D, parent(u), dim)
    return out
end

"""
    NSEBaseMPIExt.derivative_matrix(g::AbstractSquareDuctGrid, stor_dim, Val(order), Val(adj))

Return the cross-section FD matrix for storage dimension `stor_dim`, derivative
`order`, and adjoint flag `adj`.

`NSEBaseMPIExt._dd_over!` calls this on the parent (serial) grid of a
`DecomposedGrid` to obtain the local stencil matrix before applying it to each
rank's slab. Both cross-section dimensions (1 for x, 2 for y) share the same
`D₁`/`D₂` operators. Defining this method here rather than in a package
extension avoids boilerplate: `NSEBaseMPIExt` is a direct dependency and the
method is always needed for any MPI run. Without it the decomposed derivative
kernel throws a `MethodError` at runtime.
"""
function NSEBaseMPIExt.derivative_matrix(g::AbstractSquareDuctGrid,
                                          stor_dim::Int,
                                          ::Val{ORDER},
                                          ::Val{ADJ}=Val(false)) where {ORDER, ADJ}
    stor_dim in DUCT_INHOMOGENEOUS_DIMS ||
        throw(ArgumentError("storage dimension $stor_dim is not an inhomogeneous duct direction"))
    ORDER == 1 && return ADJ ? g.D₁⁺ : g.D₁
    ORDER == 2 && return ADJ ? g.D₂⁺ : g.D₂
    throw(ArgumentError("only orders 1 and 2 are available, got order=$ORDER"))
end

"""
    NSEBase.inhomogeneous_laplacian!(out, u; adjoint=false)

Apply the cross-section finite-difference Laplacian `∂²/∂x² + ∂²/∂y²` in-place.

The two terms are accumulated using the FDGrids `Val(true)` overwrite/accumulate
flag, avoiding any temporary allocation. The homogeneous (spanwise/temporal)
Fourier contribution is added separately by `NSEBase.add_homogeneous_laplacian!`.
With `adjoint=true` each second-derivative matrix is replaced by its weighted
discrete adjoint `D₂⁺`.
"""
function NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G},
                                          u::NSEBase.FTField{G};
                                          adjoint::Bool=false) where {G<:AbstractSquareDuctGrid{<:Any, <:Any, NSEBase.Undecomposed}}
    D2 = adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂
    LinearAlgebra.mul!(parent(out), D2, parent(u), Val(DUCT_INHOMOGENEOUS_DIMS[1]), Val(false))
    LinearAlgebra.mul!(parent(out), D2, parent(u), Val(DUCT_INHOMOGENEOUS_DIMS[2]), Val(true))
    return out
end
