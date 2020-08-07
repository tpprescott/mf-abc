module ElectroTaxis

using ..LikelihoodFree
using Statistics
using CSV

# Parameter values
export SingleCellModel, SpeedChange, PolarityBias, PositionBias, AlignmentBias, SingleCellBiases

# Base model
SingleCellModel = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
), NTuple{4, Float64}}
# All the additional parameters are nondimensional and correspond to presence or absence of EM field of nondimensionalised magnitude 1 only
SpeedChange = NamedTuple{(:speed_change,), Tuple{Float64}}
PolarityBias = NamedTuple{(:polarity_bias,), Tuple{Float64}}
PositionBias = NamedTuple{(:position_bias,), Tuple{Float64}}
AlignmentBias = NamedTuple{(:alignment_bias,), Tuple{Float64}}
SingleCellBiases = merge(SpeedChange, PolarityBias, PositionBias, AlignmentBias)

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

#=
function get_angles(y::AbstractArray{Complex{Float64}, 1})
    summary = Array{Float64,1}(undef, 3)
    dy = diff(y)
    arg_dy = Array{Float64, 1}(undef, size(dy))
    broadcast!(angle, arg_dy, dy)
    summary[1] = angle(y[end])
    summary[2] = mean(arg_dy)
    summary[3] = std(arg_dy)
    return summary
end
=#

export NoEF_trajectories, NoEF_displacements
const NoEF_df = CSV.read("applications/electro/No_EF.csv")
const NoEF_trajectories = collect(Iterators.partition(complex.(NoEF_df[!,:x], NoEF_df[!,:y]), 37))

const NoEF_displacements = get_displacements.(NoEF_trajectories)

export EF_trajectories, EF_displacements, EF_angles
const EF_df = CSV.read("applications/electro/With_EF.csv")
const EF_trajectories = collect(Iterators.partition(complex.(EF_df[!,:x], EF_df[!,:y]), 37))

const EF_displacements = get_displacements.(EF_trajectories)
# const EF_angles = get_angles.(EF_trajectories)

t_obs = collect(0.:5.:180.)

include("single_cell_simulator.jl")

end

# include("ElectroTaxisSingleCellAnalysis.jl")
