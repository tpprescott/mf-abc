using Distributed
@everywhere include("likelihood_free.jl")
@everywhere include("applications/electro/ElectroTaxis.jl")
using .LikelihoodFree
using .ElectroTaxis

include("applications/electro/ElectroTaxis.jl")
using .ElectroTaxisSingleCellAnalysis