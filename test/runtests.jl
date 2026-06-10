using Test
using LinearAlgebra
using Statistics
using FDGrids
using NSEBase
using ReSolverSquareDuct

# ------------------------------------------------------------------ #
# Test grid                                                           #
# ------------------------------------------------------------------ #

const N     = 64
const WIDTH = 7
const NZ    = 1   # single-mode grid: we never touch the FFT directions here
const NT    = 1
const ALPHA = 2π

const G = SquareDuctGrid(N, WIDTH, NZ, NT, ALPHA;
                         dist = FDGrids.GaussLobattoGrid())

const XS   = G.xs
const WS   = G.ws
const WS2D = G.ws2d

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

ip2d(u::AbstractMatrix, v::AbstractMatrix) = dot(WS2D .* u, v)

# Wrap a 2D cross-section array into an FTField (shape N×N×1×1)
function ftfield_from_2d(g::AbstractSquareDuctGrid,
                         u_2d::AbstractMatrix)
    u_ft = NSEBase.FTField(g)
    parent(u_ft)[:, :, 1, 1] .= u_2d
    return u_ft
end

# Apply dd! along storage dimension `dim`; return 2D cross-section result
function apply_dd(g::AbstractSquareDuctGrid,
                  u_2d::AbstractMatrix,
                  dim::Int;
                  adjoint::Bool = false)
    out = NSEBase.FTField(g)
    NSEBase.dd!(out, ftfield_from_2d(g, u_2d), Val(dim); adjoint)
    return parent(out)[:, :, 1, 1]
end

# Apply inhomogeneous_laplacian! (∂²/∂x² + ∂²/∂y²); return 2D result
function apply_lap(g::AbstractSquareDuctGrid,
                   u_2d::AbstractMatrix;
                   adjoint::Bool = false)
    out = NSEBase.FTField(g)
    NSEBase.inhomogeneous_laplacian!(out, ftfield_from_2d(g, u_2d); adjoint)
    return parent(out)[:, :, 1, 1]
end

# ------------------------------------------------------------------ #
# 1D test functions (all zero at x = 0 and x = 1 for no-slip BCs)   #
# ------------------------------------------------------------------ #

poly11(xs::AbstractVector)  = @. xs * (1 - xs)          # x(1-x)
sin2pi(xs::AbstractVector)  = @. sin(2π * xs)           # sin(2πx)
poly22(xs::AbstractVector)  = @. xs^2 * (1 - xs)^2      # x²(1-x)²

# ------------------------------------------------------------------ #
# 1. Quadrature accuracy                                              #
# ------------------------------------------------------------------ #

@testset "Quadrature accuracy" begin
    f = sin.(π .* XS)
    @test sum(WS .* f.^2)           ≈ 0.5  rtol=1e-12   # ∫₀¹ sin²(πx) dx   = 1/2
    @test sum(WS2D .* (f * f').^2)  ≈ 0.25 rtol=1e-12   # ∫∫ sin⁴ = 1/4

    @test sum(WS .* poly11(XS))                     ≈ 1/6  rtol=1e-12   # ∫₀¹ x(1-x) dx   = 1/6
    @test sum(WS2D .* (poly11(XS) * poly11(XS)'))   ≈ 1/36 rtol=1e-12   # ∫∫ x(1-x)y(1-y) = 1/36
end

# ------------------------------------------------------------------ #
# 2. dd! derivative accuracy                                          #
# ------------------------------------------------------------------ #
#
# u(x,y) = x(1-x)·y(1-y):    ∂u/∂x = (1-2x)·y(1-y)
#                               ∂u/∂y = x(1-x)·(1-2y)
# u(x,y) = sin(2πx)·y²(1-y)²: ∂u/∂x = 2π·cos(2πx)·y²(1-y)²

@testset "dd! derivative accuracy" begin
    u = poly11(XS) * poly11(XS)'       # x(1-x)·y(1-y)

    exact_x = (1 .- 2 .* XS) * poly11(XS)'
    @test maximum(abs, apply_dd(G, u, 1) - exact_x) < 1e-8

    exact_y = poly11(XS) * (1 .- 2 .* XS)'
    @test maximum(abs, apply_dd(G, u, 2) - exact_y) < 1e-8

    u_sinpoly = sin2pi(XS) * poly22(XS)'   # sin(2πx)·y²(1-y)²
    exact_sinx = (2π .* cos.(2π .* XS)) * poly22(XS)'
    @test maximum(abs, apply_dd(G, u_sinpoly, 1) - exact_sinx) < 1e-6
end

# ------------------------------------------------------------------ #
# 3. dd! adjointness                                                  #
# ------------------------------------------------------------------ #
#
# <dd!(u, dim), v>_w  ==  <u, dd!(v, dim; adjoint=true)>_w
# tested for dim = 1 (∂/∂x) and dim = 2 (∂/∂y)

@testset "dd! adjointness" begin
    # Use asymmetric polynomials (non-symmetric around x=0.5) to avoid
    # accidental cancellation: <D(symmetric), symmetric>_w = 0 for GL grids.
    poly21(x) = x.^2 .* (1 .- x)
    poly12(x) = x .* (1 .- x).^2
    pairs = [
        (poly21(XS) * poly12(XS)', poly12(XS) * poly21(XS)'),
        (poly11(XS) * poly21(XS)', poly21(XS) * poly12(XS)'),
        (poly22(XS) * poly12(XS)', poly21(XS) * poly11(XS)'),
    ]
    for (u, v) in pairs, dim in (1, 2)
        @test ip2d(apply_dd(G, u, dim), v) ≈
              ip2d(u, apply_dd(G, v, dim; adjoint = true)) rtol=1e-12
    end
end

# ------------------------------------------------------------------ #
# 4. inhomogeneous_laplacian! adjointness                            #
# ------------------------------------------------------------------ #
#
# <Δu, v>_w  ==  <u, Δ⁺v>_w   where Δ = ∂²/∂x² + ∂²/∂y²

@testset "inhomogeneous_laplacian! adjointness" begin
    pairs = [
        (poly11(XS) * sin2pi(XS)',  sin2pi(XS) * poly22(XS)'),
        (sin2pi(XS) * poly22(XS)',  poly22(XS) * poly11(XS)'),
        (poly22(XS) * poly11(XS)',  poly11(XS) * sin2pi(XS)'),
        (poly11(XS) * sin2pi(XS)',  poly11(XS) * sin2pi(XS)'),
    ]
    for (u, v) in pairs
        @test ip2d(apply_lap(G, u), v) ≈
              ip2d(u, apply_lap(G, v; adjoint = true)) rtol=1e-12
    end
end

# ------------------------------------------------------------------ #
# 5. 2D cross-section field norms (analytical)                       #
# ------------------------------------------------------------------ #
#
# u(x,y) = sin(πx)·sin(πy):
#   ||u||²      = 1/4
#   ||∂u/∂x||²  = π²/4    (= ||∂u/∂y||² by symmetry)
#   ||Δu||²     = π⁴       (Δu = -2π²u → ||Δu||² = 4π⁴·1/4)

@testset "2D field norms (analytical)" begin
    u = sin.(π .* XS) * sin.(π .* XS)'

    @test ip2d(u, u) ≈ 1/4 rtol=1e-12

    @test ip2d(apply_dd(G, u, 1), apply_dd(G, u, 1)) ≈ π^2/4 rtol=1e-8
    @test ip2d(apply_dd(G, u, 2), apply_dd(G, u, 2)) ≈ π^2/4 rtol=1e-8

    @test ip2d(apply_lap(G, u), apply_lap(G, u)) ≈ π^4 rtol=1e-6
end

# ------------------------------------------------------------------ #
# 6. NSEBase.weights interface                                        #
# ------------------------------------------------------------------ #

@testset "NSEBase.weights interface" begin
    @test NSEBase.weights(G) === WS2D
    @test size(WS2D) == (N, N)
    @test WS2D ≈ WS * WS' rtol=1e-15
end

# ------------------------------------------------------------------ #
# 7. growto preserves cross-section operators                        #
# ------------------------------------------------------------------ #

@testset "growto preserves cross-section" begin
    g2 = NSEBase.growto(G, (63, 3))
    @test g2.D₁   === G.D₁
    @test g2.D₂   === G.D₂
    @test g2.D₁⁺  === G.D₁⁺
    @test g2.D₂⁺  === G.D₂⁺
    @test g2.xs   === G.xs
    @test g2.ws   === G.ws
    @test g2.ws2d === G.ws2d
    @test size(g2) == (N, N, 63, 3)
end

# ------------------------------------------------------------------ #
# 8. 4D velocity-field norm with known analytical value              #
# ------------------------------------------------------------------ #
#
# Test function (separable, satisfies no-slip BCs in x and y):
#
#   u(x, y, z, t) = sin(πx) · sin(πy) · exp(sin(2πz)) · exp(sin(2πt))
#
# With α = 2π (Lz = 1), z ∈ [0, 1) and t ∈ [0, 1).
#
# By Parseval, with the NSEBase spectral inner product:
#
#   ||u||²_NSEBase = ½ · Σ_xy · Σ_z · Σ_t
#
# where
#   Σ_xy = ∫₀¹∫₀¹ sin²(πx)sin²(πy) dx dy  = 1/4
#   Σ_z  = ∫₀¹ exp(2sin(2πz)) dz           = I₀(2)
#   Σ_t  = ∫₀¹ exp(2sin(2πt)) dt           = I₀(2)
#
# Three independent checks:
#   (a) FFT norm ≈ Parseval direct sum (uses NSEBase.weights, machine precision)
#   (b) FFT norm ≈ ½ · Σ_xy · I₀(2)²     (analytical, fine-grid I₀(2))
#   (c) VectorField (3 identical components) norm² = 3 × single-component norm²

const G_NORM = SquareDuctGrid(N, WIDTH, 63, 63, ALPHA;
                              dist = FDGrids.GaussLobattoGrid())

@testset "4D velocity-field norm (analytical)" begin
    u_fun = (x, y, z, t) -> sin(π*x) * sin(π*y) * exp(sin(2π*z)) * exp(sin(2π*t))

    # (a) Parseval: FFT norm == direct quadrature sum -------------------
    #
    #   ||u||² = ½ · (1/(Nz·Nt)) · Σ_{ix,iy,jz,jt} ws2d[ix,iy] · u²
    #
    # Coordinate arrays and weights both from G_NORM → exact Parseval identity.

    X, Y, Z, T  = NSEBase.points(G_NORM)
    u_phys      = @. u_fun(X, Y, Z, T)
    ws2d_norm   = NSEBase.weights(G_NORM)
    norm_direct = 0.5 * sum(reshape(ws2d_norm, N, N, 1, 1) .* u_phys.^2) / (63 * 63)

    û = NSEBase.FFT(NSEBase.Field(G_NORM, u_fun))

    @test LinearAlgebra.norm(û)^2 ≈ norm_direct rtol=1e-12

    # (b) Analytical: ½ · Σ_xy · I₀(2)² -----------------------------------
    #
    # I₀(2) = ∫₀¹ exp(2sin(2πz)) dz via 100 001-point trapezoidal sum;
    # error < 1e-15 for this smooth periodic integrand.

    I0_2  = mean(exp.(2 .* sin.(2π .* (0:100_000) ./ 100_001)))
    sx    = sin.(π .* vec(X))          # sin(πx) at G_NORM collocation points
    Σ_xy  = sum(ws2d_norm .* (sx * sx').^2)
    @test LinearAlgebra.norm(û)^2 ≈ 0.5 * Σ_xy * I0_2^2 rtol=1e-10

    # (c) VectorField: three identical components → norm² = 3 × single ----
    q = NSEBase.FFT(NSEBase.VectorField(G_NORM, u_fun, u_fun, u_fun))
    @test LinearAlgebra.norm(q)^2 ≈ 3 * LinearAlgebra.norm(û)^2 rtol=1e-14
end
