module ReSolverSquareDuct

import FFTW
import FDGrids
import LinearAlgebra
import NSEBase
import NSEBaseMPIExt

export SquareDuctGrid, AbstractSquareDuctGrid
export DUCT_AXES, DUCT_FFT_ORDER, DUCT_INHOMOGENEOUS_DIMS
export SquareDuctFlow
export ConstantForcing

include("grid.jl")
include("interface.jl")
include("equations.jl")

end
