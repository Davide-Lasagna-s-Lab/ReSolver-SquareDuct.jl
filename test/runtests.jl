using Test
using LinearAlgebra
using FDGrids
using NSEBase
using ReSolverSquareDuct

# ------------------------------------------------------------------ #
# Test grid                                                           #
# ------------------------------------------------------------------ #
#
# N=64 GaussLobattoGrid with a width-7 stencil. The grid has 64 Chebyshev-
# Lobatto nodes on [0,1] including both endpoints (needed for no-slip BCs).
# Width 7 gives 6th-order accuracy for smooth functions on this grid.

const N     = 64
const WIDTH = 7
const NZ    = 1   # Nz: must be odd; a single-z-mode grid is sufficient for
const NT    = 1   # Nt: these tests — we never touch the FFT directions
const ALPHA = 2π

const G = SquareDuctGrid(N, WIDTH, NZ, NT, ALPHA;
                         dist = FDGrids.GaussLobattoGrid())

const XS = G.xs      # N collocation points on [0,1]
const WS = G.ws      # 1D quadrature weights
const WS2D = G.ws2d  # (N×N) outer-product weight matrix

# ------------------------------------------------------------------ #
# Helper: dense matrix forms                                          #
# ------------------------------------------------------------------ #

const D1  = Matrix(G.D₁)
const D2  = Matrix(G.D₂)
const D1A = Matrix(G.D₁⁺)
const D2A = Matrix(G.D₂⁺)

# 2D inner product: <u, v>_w = sum_{i,j} ws2d[i,j] * u[i,j] * conj(v[i,j])
ip2d(u, v) = dot(WS2D .* u, v)
# 1D inner product: <u, v>_w = sum_i ws[i] * u[i] * conj(v[i])
ip1d(u, v) = dot(WS .* u, v)

# ------------------------------------------------------------------ #
# Smooth test functions (satisfy homogeneous Dirichlet BCs at 0 and 1)#
# ------------------------------------------------------------------ #

const u1d_a = @. XS * (1 - XS)          # x(1-x)
const u1d_b = @. sin(2π * XS)            # sin(2πx)  (also zero at endpoints)
const u1d_c = @. XS^2 * (1 - XS)^2      # x²(1-x)²

# 2D cross-product fields (outer products of 1D functions)
const u2d_a = u1d_a * u1d_b'   # u[ix, iy] = x(1-x) * sin(2πy)
const u2d_b = u1d_b * u1d_c'   # u[ix, iy] = sin(2πx) * y²(1-y)²
const u2d_c = u1d_a * u1d_a'   # u[ix, iy] = x(1-x) * y(1-y)

# ------------------------------------------------------------------ #
# 1. Quadrature accuracy                                              #
# ------------------------------------------------------------------ #

@testset "Quadrature accuracy" begin
    # ∫₀¹ sin²(πx) dx = 1/2 → ∫₀¹∫₀¹ sin²(πx)sin²(πy) dx dy = 1/4
    f = sin.(π * XS)
    @test sum(WS .* f.^2) ≈ 0.5 rtol=1e-12

    F = f * f'
    @test sum(WS2D .* F.^2) ≈ 0.25 rtol=1e-12

    # ∫₀¹ x(1-x) dx = 1/6
    @test sum(WS .* u1d_a) ≈ 1/6 rtol=1e-12

    # ∫₀¹∫₀¹ x(1-x) y(1-y) dx dy = 1/36
    @test sum(WS2D .* u2d_c) ≈ 1/36 rtol=1e-12
end

# ------------------------------------------------------------------ #
# 2. First-derivative accuracy                                        #
# ------------------------------------------------------------------ #

@testset "D₁ derivative accuracy" begin
    # D₁ * x(1-x) should be close to 1-2x for a sufficiently wide stencil.
    # WIDTH=7 gives 6th-order accuracy; error should be far below 1e-8 at N=64.
    exact = @. 1 - 2*XS
    computed = D1 * u1d_a
    @test maximum(abs, computed - exact) < 1e-8

    # D₁ * sin(2πx) ≈ 2π cos(2πx)
    exact2 = @. 2π * cos(2π * XS)
    @test maximum(abs, D1 * u1d_b - exact2) < 1e-8

    # D₁ * x²(1-x)² ≈ 2x(1-x)² - 2x²(1-x)
    exact3 = @. 2*XS*(1-XS)^2 - 2*XS^2*(1-XS)
    @test maximum(abs, D1 * u1d_c - exact3) < 1e-8
end

# ------------------------------------------------------------------ #
# 3. First-derivative adjointness                                     #
# ------------------------------------------------------------------ #

@testset "D₁⁺ adjointness" begin
    # <D₁ u, v>_w == <u, D₁⁺ v>_w  for all smooth u, v
    for (u, v) in [(u1d_a, u1d_b), (u1d_b, u1d_c), (u1d_c, u1d_a),
                   (u1d_a, u1d_a), (u1d_b, u1d_b)]
        lhs = ip1d(D1 * u, v)
        rhs = ip1d(u, D1A * v)
        @test lhs ≈ rhs rtol=1e-12
    end
end

# ------------------------------------------------------------------ #
# 4. Second-derivative adjointness                                    #
# ------------------------------------------------------------------ #

@testset "D₂⁺ adjointness" begin
    # <D₂ u, v>_w == <u, D₂⁺ v>_w
    for (u, v) in [(u1d_a, u1d_b), (u1d_b, u1d_c), (u1d_c, u1d_a),
                   (u1d_a, u1d_a), (u1d_b, u1d_b)]
        lhs = ip1d(D2 * u, v)
        rhs = ip1d(u, D2A * v)
        @test lhs ≈ rhs rtol=1e-12
    end
end

# ------------------------------------------------------------------ #
# 5. 2D cross-section Laplacian adjointness                          #
# ------------------------------------------------------------------ #
#
# The 2D Laplacian Δ = ∂²/∂x² + ∂²/∂y² acts on N×N arrays.
# Applying D₂ along dim 1 (x): Δx u = D2 * u
# Applying D₂ along dim 2 (y): Δy u = u * D2'
#
# Adjoint operators: D2A along the respective dim.
# <Δu, v>_{ws2d} == <u, Δ⁺v>_{ws2d}

@testset "2D Laplacian adjointness" begin
    lap(u)  = D2  * u .+ u * D2'
    lapA(v) = D2A * v .+ v * D2A'

    for (u, v) in [(u2d_a, u2d_b), (u2d_b, u2d_c), (u2d_c, u2d_a),
                   (u2d_a, u2d_a)]
        lhs = ip2d(lap(u),  v)
        rhs = ip2d(u, lapA(v))
        @test lhs ≈ rhs rtol=1e-12
    end
end

# ------------------------------------------------------------------ #
# 6. 2D norm of a known analytical field                             #
# ------------------------------------------------------------------ #
#
# For u(x,y) = sin(πx) sin(πy) on [0,1]²:
#   ||u||² = ∫₀¹∫₀¹ sin²(πx)sin²(πy) dx dy = 1/4
#
# For ∇u applied by D₁:
#   ||∂u/∂x||² = ∫₀¹∫₀¹ π²cos²(πx)sin²(πy) dx dy = π²/4

@testset "2D field norm (analytical)" begin
    sinx = sin.(π .* XS)
    cosx = cos.(π .* XS)
    u    = sinx * sinx'

    @test ip2d(u, u) ≈ 1/4 rtol=1e-12

    dux = D1 * u   # ∂u/∂x applied along dim 1
    @test ip2d(dux, dux) ≈ π^2/4 rtol=1e-8

    duy = u * D1'  # ∂u/∂y applied along dim 2
    @test ip2d(duy, duy) ≈ π^2/4 rtol=1e-8

    # ||Δu||² = ||-2π² sin(πx)sin(πy)||² = 4π⁴ * 1/4 = π⁴
    lpu = D2 * u .+ u * D2'
    @test ip2d(lpu, lpu) ≈ π^4 rtol=1e-6
end

# ------------------------------------------------------------------ #
# 7. weights(g) returns ws2d                                         #
# ------------------------------------------------------------------ #

@testset "NSEBase.weights interface" begin
    @test NSEBase.weights(G) === WS2D
    @test size(WS2D) == (N, N)
    # ws2d is a rank-1 outer product: row i is ws[i] times the full ws vector
    @test WS2D ≈ WS * WS' rtol=1e-15
end

# ------------------------------------------------------------------ #
# 8. growto preserves operators                                       #
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
