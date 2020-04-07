module ElectroTaxis

using ..LikelihoodFree
using ..ABC
using Statistics

# Parameter values
export SingleCellModel_NoEF
SingleCellModel_NoEF = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
), NTuple{4, Float64}}

# Experiment and data
export NoEF_Experiment
struct NoEF_Experiment{Y} <: AbstractExperiment{Y}
    y_obs::Y
    t_obs::AbstractArray{Float64,1}
end
include("output_spaces.jl")    
include("single_cell_simulator.jl")

using .SingleCell
export SingleCellSimulator_NoEF


end