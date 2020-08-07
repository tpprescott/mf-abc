module ElectroTaxis

using ..LikelihoodFree
using Statistics
using CSV

# Parameter values
export SingleCellModel, VelocityBias, SpeedIncrease, SpeedAlignment, PolarityBias, ElectroEffects

# Base model
SingleCellModel = NamedTuple{(
    :v,
    :EB_on,
    :EB_off,
    :D,
), NTuple{4, Float64}}
# All the additional parameters are nondimensional and correspond to presence or absence of EM field of nondimensionalised magnitude 1 only
VelocityBias = NamedTuple{(:γ1,), Tuple{Float64}}
SpeedIncrease = NamedTuple{(:γ2,), Tuple{Float64}}
SpeedAlignment = NamedTuple{(:γ3,), Tuple{Float64}}
PolarityBias = NamedTuple{(:γ4,), Tuple{Float64}}
ElectroEffects = merge(VelocityBias, SpeedIncrease, SpeedAlignment, PolarityBias)

function get_displacements(y::AbstractArray{Complex{Float64}, 1})
    summary = Array{Float64,1}(undef, 4)
    dy = diff(y)
    abs_dy = Array{Float64,1}(undef, size(dy))
    broadcast!(abs, abs_dy, dy)
    summary[1] = abs(y[end])
    summary[2] = mean(abs_dy)
    summary[3] = std(abs_dy)
    summary[4] = angle(y[end])
    return summary
end

export NoEF_trajectories, NoEF_displacements
const NoEF_df = CSV.read("applications/electro/No_EF.csv")
const NoEF_trajectories = collect(Iterators.partition(complex.(NoEF_df[!,:x], NoEF_df[!,:y]), 37))

const NoEF_displacements = get_displacements.(NoEF_trajectories)

export EF_trajectories, EF_displacements, EF_angles
const EF_df = CSV.read("applications/electro/With_EF.csv")
const EF_trajectories = collect(Iterators.partition(complex.(EF_df[!,:x], EF_df[!,:y]), 37))

const EF_displacements = get_displacements.(EF_trajectories)

t_obs = collect(0.:5.:180.)

include("single_cell_simulator.jl")

end
