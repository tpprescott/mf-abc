module SyntheticBayes
using ..LikelihoodFree

using Distributions
using Statistics
# MvNormal(μ::Float64, σ2::Float64) = Normal(μ, sqrt(σ2))

export SyntheticLikelihood, SyntheticBayesWeight
struct SyntheticLikelihood <: AbstractComparison end

SyntheticBayesWeight{Simulator} = LikelihoodFreeWeight{Simulator, SyntheticLikelihood}
SyntheticBayesWeight(F::Simulator, K::Int64=1) where Simulator = LikelihoodFreeWeight(F, SyntheticLikelihood(), K)

function (C::SyntheticLikelihood)(y_obs::Array{Array{T,1},1}, y::Array{Array{T,1},1}; kwargs...) where T

    length(y)//length(y_obs) >= 10 || (@warn "Only $(length(y)) simulations for $(length(y_obs)) observations.")

    μ = mean(y)
    Σ = cov(y)
    sb_lh = MvNormal(μ, Σ)
    logw_vec = logpdf.(Ref(sb_lh), y_obs)
    logw = sum(logw_vec)
    w = exp(logw)

    return (w=w, logw=logw, sb_lh=sb_lh)
end

end