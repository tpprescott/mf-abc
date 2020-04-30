#######
# Generate models from the generator

# Generics for AbstractGenerator

import Distributions.rand, Distributions.rand!
import Distributions.logpdf, Distributions.logpdf!
export rand, rand!, logpdf, logpdf!

# Basic rand is to produce θ::Θ - need to implement (q::AbstractGenerator::M)(; kwargs...)
function rand(q::AbstractGenerator{Θ}; prior::AbstractGenerator{Θ}=q, kwargs...)::NamedTuple{(:θ, :logq, :logp), Tuple{Θ, Float64, Float64}} where Θ<:AbstractModel
    return q(; prior=prior, kwargs...)
end
function logpdf(q::AbstractGenerator{Θ}, θ::Θ) where Θ
    error("Implement logpdf for $(typeof(q)), returning a named tuple")
end

include("generate_parameters_distributions.jl")
include("generate_parameters_perturbations.jl")
# include("generate_parameters_smc.jl")