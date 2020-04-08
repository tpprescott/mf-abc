module SyntheticBayes
using ..LikelihoodFree

using Distributions
import Distributions.MvNormal
MvNormal(μ::Float64, σ2::Float64) = Normal(μ, sqrt(σ2))

export SyntheticLikelihood

struct SyntheticLikelihood{U,Y} <: AbstractComparison{U,Y} end

import ..LikelihoodFree.compare, ..LikelihoodFree.compare!
export compare, compare!
function compare(c::SyntheticLikelihood{U,Y}, u::U, y::AbstractArray{Y,1})::NamedTuple where U where Y
    K = length(y)
    dim = length(first(y))

    K>1 || error("Need more than one simulation at each parameter value for synthetic Bayes")
    K>20 || (@warn "Only $K simulations for each parameter value")

    # Define approximate likelihood function from simulations
    μ = mean(y)
    Σ = cov(y) # <-- This could be more sophisticated (at least in the multidimensional case)
    
    if dim == 1
        sb_lh = Normal(μ, sqrt(Σ))
        logw = sum(logpdf.(sb_lh, u.y_obs))
    else
        sb_lh = MvNormal(μ, Σ)
        logw = sum(logpdf(sb_lh, u.y_obs))
    end

    # Evaluate approximate log-likelihood
    return (w = exp(logw), logw = logw, sb_lh = sb_lh)
end
function compare!(ww::Array{Float64, 1}, c::SyntheticLikelihood, u::U, yy::AbstractArray{Y,2}) where U where Y
    dim = length(first(yy))
    
    logww = Array{Float64,1}(undef, length(ww))
    sb_lh = dim==1 ? Array{Normal, 1}(undef, length(ww)) : Array{MvNormal, 1}(undef, length(ww))
    for i in eachindex(ww)
        comparison = compare(c, u, view(yy,i,:))
        ww[i] = comparison[:w]
        logww[i] = comparison[:logw]
        sb_lh[i] = comparison[:sb_lh]
    end
    return (logww=logww, sb_lh=sb_lh)
end
function compare(c::SyntheticLikelihood, u::U, yy::AbstractArray{Y,2}) where U where Y
    ww = Array{Float64,1}(undef, size(yy,1))
    save = compare!(ww, c, u ,yy)
    return merge((ww=ww,), save)
end

end