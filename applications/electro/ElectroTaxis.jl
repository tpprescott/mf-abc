module ElectroTaxis

using ..LikelihoodFree
using ..ABC

# Parameter values
export SingleCellModel_NoEF
SingleCellModel_NoEF = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
), NTuple{4, Float64}}

# Experiment and data
export SingleCellTrajectories_NoEF
struct SingleCellTrajectories_NoEF <: AbstractExperiment{Array{Float64,2}}
    y_obs::Array{Float64,2} # Data consists of all positions in times of t_obs
    t_obs::Array{Float64}
end

include("single_cell_simulator.jl")
using .SingleCell
export SingleCellSimulator_NoEF

## Output space
export MatchedTimeSeries, TSFreshOutput
struct MatchedTimeSeries <: AbstractSummaryStatisticSpace
    y::Array{Complex{Float64},2}
end
struct TSFreshOutput <: AbstractSummaryStatisticSpace
    y::Array{Float64,2}
end
# Need to define TSFreshOutput(y::RODESolution)

end