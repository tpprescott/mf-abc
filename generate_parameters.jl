#######
# Generate models from the generator

# Generics for AbstractGenerator

import Distributions.rand, Distributions.rand!
export rand

function rand(q::AbstractGenerator{M})::NamedTuple where M<:AbstractModel
    return (m=q(),)
end
function rand!(mm::AbstractArray{M}, q::AbstractGenerator{M})::NamedTuple where M<:AbstractModel
    for i in eachindex(mm)
        mm[i] = rand(q)[:m]
    end
    return NamedTuple()
end
function rand(q::AbstractGenerator{M}, N::Int64)::NamedTuple where M<:AbstractModel
    mm = Array{M,1}(undef, N)
    save = rand!(mm, q)
    return merge((mm=mm,), save)
end

function unnormalised_importance_weight(q::AbstractGenerator{M}, m::M)::NamedTuple where M<:AbstractModel
    return (p=q(m),)
end
function unnormalised_importance_weight!(pp::AbstractArray{M}, q::AbstractGenerator{M}, mm::AbstractArray{M})::NamedTuple where M<:AbstractModel
    for i in eachindex(mm)
        pp[i] = unnormalised_importance_weight(q, mm[i])[:p]
    end
    return NamedTuple()
end
function unnormalised_importance_weight(q::AbstractGenerator{M}, mm::AbstractArray{M})::NamedTuple where M<:AbstractModel
    pp = Array{Float64}(undef, size(mm))
    save = unnormalised_importance_weight!(pp, q, mm)
    return merge((pp=pp,), save)
end

include("generate_parameters_distributions.jl")
include("generate_parameters_smc.jl")