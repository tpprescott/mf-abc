module ElectroTaxis

using ..LikelihoodFree
using Statistics
using CSV

# Parameter values
export SingleCellModel_NoEF, SingleCellModel_EF, SingleCellModel
SingleCellModel_NoEF = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
), NTuple{4, Float64}}

SingleCellModel_EF = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
    :EF_bias,
), NTuple{5, Float64}}
SingleCellModel = Union{SingleCellModel_EF, SingleCellModel_NoEF}

function get_displacements(y::AbstractArray{Complex{Float64}, 1})
    summary = Array{Float64,1}(undef, 3)
    dy = diff(y)
    abs_dy = Array{Float64,1}(undef, size(dy))
    broadcast!(abs, abs_dy, dy)
    summary[1] = abs(y[end])
    summary[2] = mean(abs_dy)
    summary[3] = std(abs_dy)
    return summary
end

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
# cosarg(x) = cos(angle(x))


export NoEF_trajectories, NoEF_displacements
const NoEF_df = CSV.read("applications/electro/No_EF.csv")
const NoEF_trajectories = collect(Iterators.partition(complex.(NoEF_df[!,:x], NoEF_df[!,:y]), 37))

const NoEF_displacements = get_displacements.(NoEF_trajectories)

export EF_trajectories, EF_displacements, EF_angles
const EF_df = CSV.read("applications/electro/With_EF.csv")
const EF_trajectories = collect(Iterators.partition(complex.(EF_df[!,:x], EF_df[!,:y]), 37))

const EF_displacements = get_displacements.(EF_trajectories)
const EF_angles = get_angles.(EF_trajectories)

t_obs = collect(0.:5.:180.)

include("single_cell_simulator.jl")

end

include("ElectroTaxisSingleCellAnalysis.jl")
