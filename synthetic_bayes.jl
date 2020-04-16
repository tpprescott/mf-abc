module SyntheticBayes
using ..LikelihoodFree

using Distributions
using Statistics
# MvNormal(μ::Float64, σ2::Float64) = Normal(μ, sqrt(σ2))

export SyntheticLikelihood
struct SyntheticLikelihood{T} <: AbstractComparison{T} end

import ..LikelihoodFree.compare, ..LikelihoodFree.compare!
export compare, compare!

function μ!(μ::AbstractArray{T,2}, yy::AbstractArray{T,4}) where T
    (dim, numParameters) = (size(yy,1), size(yy,2))
    size(μ)==(dim, numParameters) || error("Dimensions of mean and simulations don't match")
    yy_collect = reshape(yy, dim, numParameters, :)

    for nParameter in 1:numParameters
        mean!(view(μ, :, nParameter), view(yy_collect, :, nParameter, :))
    end
    return nothing
end
function Σ!(Σ::AbstractArray{T,3}, yy::AbstractArray{T,4}) where T
    (dim, numParameters) = (size(yy,1), size(yy,2))
    size(Σ)==(dim, dim, numParameters) || error("Dimensions of covariance and simulations don't match")
    yy_collect = reshape(yy, dim, numParameters, :)

    for nParameter in 1:numParameters
        view(Σ, :, :, nParameter) .= cov(view(yy_collect, :, nParameter, :), dims=2)
    end
    return nothing
end

function compare(c::SyntheticLikelihood{T}, yy::AbstractArray{T,2}, y_obs)::NamedTuple where T
    μ = mean(yy, dims=2)
    Σ = cov(yy, dims=2)
    sb_lh = MvNormal(vec(μ), Σ)
    logw = sum(logpdf(sb_lh, y_obs))
    w = exp(logw)
    return (w=w, logw=logw, sb_lh=sb_lh)
end

function compare(c::SyntheticLikelihood{T}, yy::AbstractArray{T,4}, y_obs)::NamedTuple where T
    if size(yy,2)==1
        yy_collect = reshape(yy, size(yy,1), :)
        return compare(c, yy_collect, y_obs)
    else
        ww = Array{Float64,1}(undef, size(yy,2))
        out = compare!(ww, c, yy, y_obs)
        return merge((ww=ww,), out)
    end
end

function compare!(ww::AbstractArray{Float64, 1}, c::SyntheticLikelihood{T}, yy::AbstractArray{T,4}, y_obs)::NamedTuple where T
    
    dim = size(yy,1)
    numParameters = size(yy,2)

    K = size(yy,3)*size(yy,4)

    K>1 || error("Need more than one simulation at each parameter value for synthetic Bayes")
    K>20 || (@warn "Only $K simulations for each parameter value")

    dim == size(y_obs,1) || error("Mismatched dimensions of data space and simulation output space")

    μ = Array{T,2}(undef, dim, numParameters)
    Σ = Array{T,3}(undef, dim, dim, numParameters)
    μ!(μ, yy)
    Σ!(Σ, yy)

    sb_lh = Array{MvNormal,1}(undef, numParameters)
    logww = Array{Float64, 1}(undef, numParameters)

    for nParameter in axes(yy,2)
        sb_lh[nParameter] = MvNormal(μ[:, nParameter], Σ[:,:,nParameter])
        logww[nParameter] = sum(logpdf(sb_lh[nParameter], y_obs))
        ww[nParameter] = exp(logww[nParameter])
    end
    return (logww = logww, sb_lh = sb_lh)
end


end