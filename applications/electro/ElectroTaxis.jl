module ElectroTaxis

using ..LikelihoodFree
using ..ABC
using ..SyntheticBayes
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

function get_displacements(y::AbstractArray{Complex{Float64}, 2})
    n = size(y,2)
    out = Array{Float64,2}(undef, 3, size(y,2))
    for i = 1:n
        get_displacements!(view(out,:,i), view(y,:,i))
    end
    return out
end
function get_displacements!(ss::AbstractArray{Float64, 1}, y::AbstractArray{Complex{Float64}, 1})
    dy = diff(y)
    abs_dy = Array{Float64,1}(undef, size(dy))
    broadcast!(abs, abs_dy, dy)
    ss[1] = abs(y[end])
    ss[2] = mean(abs_dy)
    ss[3] = std(abs_dy)
    return nothing
end


function get_angles(y::AbstractArray{Complex{Float64}, 2})
    n = size(y,2)
    out = Array{Float64,2}(undef, 3, size(y,2))
    for i = 1:n
        get_angles!(view(out,:,i), view(y,:,i))
    end
    return out
end
function get_angles!(ss::AbstractArray{Float64, 1}, y::AbstractArray{Complex{Float64}, 1})
    dy = diff(y)
    cosarg_dy = Array{Float64, 1}(undef, size(dy))
    broadcast!(cosarg, cosarg_dy, dy)
    ss[1] = cosarg(y[end])
    ss[2] = mean(cosarg_dy)
    ss[3] = std(cosarg_dy)
    return nothing
end
cosarg(x) = cos(angle(x))


export NoEF_trajectories, NoEF_displacements
const NoEF_df = CSV.read("applications/electro/No_EF.csv")
const NoEF_trajectories = reshape(complex.(NoEF_df[!,:x], NoEF_df[!,:y]), 37, 50)
const NoEF_displacements = get_displacements(NoEF_trajectories)

export EF_trajectories, EF_displacements, EF_angles
const EF_df = CSV.read("applications/electro/With_EF.csv")
const EF_trajectories = reshape(complex.(EF_df[!,:x], EF_df[!,:y]), 37, 50)
const EF_displacements = get_displacements(EF_trajectories)
const EF_angles = get_angles(EF_trajectories)

t_obs = collect(0.:5.:180.)

include("single_cell_simulator.jl")

end