module ElectroTaxis

using ..LikelihoodFree
using ..ABC
using ..SyntheticBayes
using Statistics
using CSV

# Parameter values
export SingleCellModel_NoEF
SingleCellModel_NoEF = NamedTuple{(
    :polarised_speed,
    :EB_on,
    :EB_off,
    :σ,
), NTuple{4, Float64}}

function get_displacements(y::AbstractArray{Complex{Float64}, 2})
    n = size(y,2)
    out = Array{Float64,2}(undef, 3, size(y,2))
    for i = 1:n
        get_displacements!(view(out,:,i), view(y,:,i))
    end
    return out
end
function get_displacements!(ss::AbstractArray{Float64, 1}, y::AbstractArray{Complex{Float64}, 1})
    displacements = diff(y)
    ss[1] = abs(y[end])
    ss[2] = mean(abs, displacements)
    ss[3] = sqrt(mean(abs2, displacements) - ss[2]^2)
    return nothing
end

export NoEF_trajectories, NoEF_displacements
const NoEF_df = CSV.read("applications/electro/No_EF.csv")
const NoEF_trajectories = reshape(complex.(NoEF_df[!,:x], NoEF_df[!,:y]), 37, 50)
const NoEF_displacements = get_displacements(NoEF_trajectories)

t_obs = collect(0.:5.:180.)

include("single_cell_simulator.jl")

end