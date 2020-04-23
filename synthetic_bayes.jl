using Distributions
using Statistics
# MvNormal(μ::Float64, σ2::Float64) = Normal(μ, sqrt(σ2))

export BayesianSyntheticLikelihood
struct BayesianSyntheticLikelihood{F, Θ<:Union{AbstractModel, Nothing}, SimOutput<:Union{NamedTuple, Nothing}} <: LikelihoodFreeLikelihoodFunction{F}
    f::F
    K::Int64
    θ::Θ
    y::SimOutput
    function BayesianSyntheticLikelihood(f::F; num_simulations=100) where F
        return new{F, Nothing, Nothing}(f, num_simulations, nothing, nothing)
    end
    function BayesianSyntheticLikelihood(f::F, θ::Θ, y::SimOutput) where F where Θ where SimOutput
        return new{F, Θ, SimOutput}(f, length(y.y), θ, y)
    end
    function BayesianSyntheticLikelihood(L::BayesianSyntheticLikelihood{F}, θ::Θ, y::SimOutput) where F where Θ where SimOutput
        return new{F, Θ, SimOutput}(L.f, length(y.y), θ, y)
    end
end

function (L::BayesianSyntheticLikelihood)(θ::AbstractModel, args...; kwargs...)
    sims = simulate(L.f, L.K; θ=θ, kwargs...)
    return BayesianSyntheticLikelihood(L, θ, sims)
end

function (L::BayesianSyntheticLikelihood)(y_obs::Array{Array{T,1},1}; loglikelihood::Bool, kwargs...) where T

    y::Array{Array{T,1}} = L.y.y
    length(y)//length(y_obs) >= 10 || (@warn "Only $(length(y)) simulations for $(length(y_obs)) observations.")

    μ = mean(y)
    Σ = cov(y)
    sb_lh = MvNormal(μ, Σ)

    logw = sum(broadcast(logpdf, Ref(sb_lh), y_obs))
    return loglikelihood ? logw : exp(logw)
end