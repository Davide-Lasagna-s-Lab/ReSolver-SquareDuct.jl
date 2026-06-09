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

ip2d(u, v) = dot(WS2D .* u, v)

# Wrap a 2D cross-section array into an FTField (shape N×N×1×1)
function ftfield_from_2d(g, u_2d)
    u_ft = NSEBase.FTField(g)
    parent(u_ft)[:, :, 1, 1] .= u_2d
    return u_ft
end

# Apply dd! along storage dimension `dim`; return 2D cross-section result
function apply_dd(g, u_2d, dim; adjoint = false)
    out = NSEBase.FTField(g)
    NSEBase.dd!(out, ftfield_from_2d(g, u_2d), Val(dim); adjoint = adjoint)
    return parent(out)[:, :, 1, 1]
end

# Apply inhomogeneous_laplacian! (∂²/∂x² + ∂²/∂y²); return 2D result
function apply_lap(g, u_2d; adjoint = false)
    out = NSEBase.FTField(g)
    NSEBase.inhomogeneous_laplacian!(out, ftfield_from_2d(g, u_2d); adjoint = adjoint)
    return parent(out)[:, :, 1, 1]
end

# ------------------------------------------------------------------ #
# Smooth test functions (zero at x = 0 and x = 1 for no-slip BCs)   #
# ------------------------------------------------------------------ #

const u1d_a = @. XS * (1 - XS)       # x(1-x)
const u1d_b = @. sin(2π * XS)         # sin(2πx)
const u1d_c = @. XS^2 * (1 - XS)^2   # x²(1-x)²

const u2d_a = u1d_a * u1d_b'   # x(1-x) · sin(2πy)
const u2d_b = u1d_b * u1d_c'   # sin(2πx) · y²(1-y)²
const u2d_c = u1d_a * u1d_a'   # x(1-x) · y(1-y)

# ------------------------------------------------------------------ #
# 1. Quadrature accuracy                                              #
# ------------------------------------------------------------------ #

@testset "Quadrature accuracy" begin
    f = sin.(π * XS)
    @test sum(WS .* f.^2) ≈ 0.5      rtol=1e-12   # ∫₀¹ sin²(πx) dx = 1/2
    @test sum(WS2D .* (f * f').^2) ≈ 0.25 rtol=1e-12   # ∫₀¹∫₀¹ sin⁴ ... = 1/4

    @test sum(WS .* u1d_a)    ≈ 1/6  rtol=1e-12   # ∫₀¹ x(1-x) dx  = 1/6
    @test sum(WS2D .* u2d_c)  ≈ 1/36 rtol=1e-12   # ∫∫ x(1-x)y(1-y) = 1/36
end

# ------------------------------------------------------------------ #
# 2. dd! derivative accuracy                                          #
# ------------------------------------------------------------------ #
#
# u(x,y) = x(1-x)·y(1-y):  ∂u/∂x = (1-2x)·y(1-y)
#                             ∂u/∂y = x(1-x)·(1-2y)
#
# u(x,y) = sin(2πx)·y²(1-y)²:  ∂u/∂x = 2π·cos(2πx)·y²(1-y)²

@testset "dd! derivative accuracy" begin
    exact_x = (1 .- 2 .* XS) * u1d_a'
    @test maximum(abs, apply_dd(G, u2d_c, 1) - exact_x) < 1e-8

    exact_y = u1d_a * (1 .- 2 .* XS)'
    @test maximum(abs, apply_dd(G, u2d_c, 2) - exact_y) < 1e-8

    exact_sinx = (2π .* cos.(2π .* XS)) * u1d_c'
    @test maximum(abs, apply_dd(G, u2d_b, 1) - exact_sinx) < 1e-8
end

# ------------------------------------------------------------------ #
# 3. dd! adjointness                                                  #
# ------------------------------------------------------------------ #
#
# <dd!(u, dim), v>_w  ==  <u, dd!(v, dim; adjoint=true)>_w
# tested for dim = 1 (∂/∂x) and dim = 2 (∂/∂y)

@testset "dd! adjointness" begin
    for (u, v) in [(u2d_a, u2d_b), (u2d_b, u2d_c), (u2d_c, u2d_a),
                   (u2d_a, u2d_a), (u2d_b, u2d_b)]
        for dim in (1, 2)
            @test ip2d(apply_dd(G, u, dim), v) ≈
                  ip2d(u, apply_dd(G, v, dim; adjoint = true)) rtol=1e-12
        end
    end
end

# ------------------------------------------------------------------ #
# 4. inhomogeneous_laplacian! adjointness                            #
# ------------------------------------------------------------------ #
#
# <Δu, v>_w  ==  <u, Δ⁺v>_w   where Δ = ∂²/∂x² + ∂²/∂y²

@testset "inhomogeneous_laplacian! adjointness" begin
    for (u, v) in [(u2d_a, u2d_b), (u2d_b, u2d_c), (u2d_c, u2d_a),
                   (u2d_a, u2d_a)]
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
    sinx = sin.(π .* XS)
    u    = sinx * sinx'

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

const NZ_NORM = 63
const NT_NORM = 63

const G_NORM = SquareDuctGrid(N, WIDTH, NZ_NORM, NT_NORM, ALPHA;
                              dist = FDGrids.GaussLobattoGrid())

@testset "4D velocity-field norm (analytical)" begin
    u_fun = (x, y, z, t) -> sin(π*x) * sin(π*y) * exp(sin(2π*z)) * exp(sin(2π*t))

    # (a) Parseval: FFT norm == direct quadrature sum -------------------
    #
    #   ||u||² = ½ · (1/(Nz·Nt)) · Σ_{ix,iy,jz,jt} ws2d[ix,iy] · u(ix,iy,jz,jt)²
    #
    # Using coordinate arrays and weights from G_NORM itself so that the
    # comparison is the exact Parseval identity (no approximation beyond
    # floating-point rounding).

    X, Y, Z, T   = NSEBase.points(G_NORM)
    u_phys       = @. u_fun(X, Y, Z, T)   # shape (N, N, NZ_NORM, NT_NORM)
    ws2d_norm    = NSEBase.weights(G_NORM)
    norm_direct  = 0.5 * sum(reshape(ws2d_norm, N, N, 1, 1) .* u_phys.^2) /
                   (NZ_NORM * NT_NORM)

    û = NSEBase.FFT(NSEBase.Field(G_NORM, u_fun))

    @test LinearAlgebra.norm(û)^2 ≈ norm_direct rtol=1e-12

    # (b) Analytical: ½ · Σ_xy · I₀(2)² -----------------------------------
    #
    # I₀(2) = ∫₀¹ exp(2sin(2πz)) dz, computed from a 100 001-point trapezoidal
    # sum. Error is below 1e-15 for this smooth periodic integrand.

    z_fine = (0:100_000) ./ 100_001
    I0_2   = mean(exp.(2 .* sin.(2π .* z_fine)))

    xs_norm = vec(X)                                   # (N,) x-coordinates from G_NORM
    sx      = sin.(π .* xs_norm)                       # (N,)
    Σ_xy    = sum(ws2d_norm .* (sx * sx').^2)          # ≈ 1/4 by quadrature
    norm_analytic = 0.5 * Σ_xy * I0_2^2

    @test LinearAlgebra.norm(û)^2 ≈ norm_analytic rtol=1e-10

    # (c) VectorField: three identical components → norm² = 3 × single ----
    q = NSEBase.FFT(NSEBase.VectorField(G_NORM, u_fun, u_fun, u_fun))
    @test LinearAlgebra.norm(q)^2 ≈ 3 * LinearAlgebra.norm(û)^2 rtol=1e-14
end
