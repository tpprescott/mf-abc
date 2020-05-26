using Distributions
using Statistics
# MvNormal(μ::Float64, σ2::Float64) = Normal(μ, sqrt(σ2))

export BayesianSyntheticLikelihood
struct BayesianSyntheticLikelihood{F, Y<:NamedTuple, SimOptions<:NamedTuple} <: LikelihoodFreeLikelihoodFunction{F}
    f::F
    y::Array{Y,1}
    kw::SimOptions

    function BayesianSyntheticLikelihood(f::F, y::Array{Y,1}; numReplicates::Int64=length(y), kwargs...) where F<:AbstractSimulator where Y
        kw = merge((numReplicates=numReplicates,), kwargs)
        return new{F, eltype(F), typeof(kw)}(f, y, kw)
    end
end

function BayesianSyntheticLikelihood(f::F; numReplicates::Int64, kwargs...) where F <: AbstractSimulator
    Y = eltype(F)
    return BayesianSyntheticLikelihood(f, Array{Y,1}(); numReplicates=numReplicates, kwargs...)
end
function BayesianSyntheticLikelihood(L::BayesianSyntheticLikelihood{F}, y::Array{Y,1}; kwargs...) where F where Y
    return BayesianSyntheticLikelihood(L.f, y; L.kw..., kwargs...)
end

function (L::BayesianSyntheticLikelihood)(θ::AbstractModel, args...; kwargs...)
    y = simulate(L.f, θ, args...; L.kw..., kwargs...)
    return BayesianSyntheticLikelihood(L, y; kwargs...)
end

function (L::BayesianSyntheticLikelihood)(y_obs; loglikelihood::Bool=true, kwargs...)

    t = table(L.y)
    y = select(t, :y) # Requires output of simulation for comparison with data to be called "y"
    length(y)//length(y_obs) >= 10 || (@warn "Only $(length(y)) simulations for $(length(y_obs)) observations." maxlog=1)

    μ = mean(y)
    Σ = cov(y)
    sb_lh = MvNormal(μ, Σ)

    logw = sum(logpdf.(Ref(sb_lh), y_obs))
    return loglikelihood ? logw : exp(logw)
end