#######
# Generate models from the generator

# Generics for AbstractGenerator

import Distributions.rand, Distributions.rand!
import Distributions.logpdf, Distributions.logpdf!
export rand, rand!, logpdf, logpdf!

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

function logpdf(q::AbstractGenerator{M}, m::M) where M
    error("Implement logpdf for $(typeof(q)), returning a named tuple")
end
function logpdf!(logqq::AbstractArray{Float64,1}, q::AbstractGenerator{M}, mm::AbstractArray{M,1}) where M
    for i in eachindex(mm)
        logqq[i] = logpdf(q, mm[i])[:logq]
    end
    return NamedTuple()
end
function logpdf(q::AbstractGenerator{M}, mm::AbstractArray{M,1}) where M
    logqq = Array{Float64,1}(undef, length(mm))
    save = logpdf!(logqq, q, mm)
    return merge((logqq=logqq, ), save)
end

include("generate_parameters_distributions.jl")
include("generate_parameters_perturbations.jl")
include("generate_parameters_smc.jl")