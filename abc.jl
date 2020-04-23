using Statistics: mean
using Distances

export ABCLikelihood
struct ABCLikelihood{F<:AbstractSimulator, D<:Metric, Θ<:Union{AbstractModel, Nothing}, SimOutput<:Union{NamedTuple, Nothing}} <: LikelihoodFreeLikelihoodFunction{F}
    f::F
    K::Int64
    distance::D
    ϵ::Float64
    θ::Θ
    y::SimOutput
    function ABCLikelihood(f::F, distance::D, ϵ::Float64; num_simulations::Int64=1) where F where D
        return new{F, D, Nothing, Nothing}(f, num_simulations, distance, ϵ, nothing, nothing)
    end
    function ABCLikelihood(f::F, distance::D, ϵ::Float64, θ::Θ, y::SimOutput) where F where D where Θ where SimOutput
        return new{F, D, Θ, SimOutput}(f, length(y.y), distance, ϵ, θ, y)
    end
    function ABCLikelihood(L::ABCLikelihood{F, D}, θ::Θ, y::SimOutput) where F where D where Θ where SimOutput
        return new{F, D, Θ, SimOutput}(L.f, length(y.y), L.distance, L.ϵ, θ, y)
    end
end
function (L::ABCLikelihood)(θ::AbstractModel, args...; kwargs...)
    sims = simulate(L.f, L.K; θ=θ, kwargs...)
    return ABCLikelihood(L, θ, sims)
end

export accept_reject

function (L::ABCLikelihood)(y_obs::Array{Array{T,1},1}; loglikelihood::Bool, kwargs...) where T
    y::Array{Array{T,1},1} = L.y.y
    isinteger(length(y)//length(y_obs)) || error("Need number of simulations to be an integer multiple of length(y_obs) = $(length(y_obs)).")
    
    y_rs = reshape(y, length(y_obs), :)
    dist = evaluate.(Ref(L.distance), y_obs, y_rs)
    compare_out::NamedTuple = accept_reject(L.ϵ, dist)
    return loglikelihood ? compare_out[:logw] : compare_out[:w]
end

_ar(epsilon, x) = Float64(x<epsilon)
function accept_reject(epsilon, d) where T 
    ww = _ar.(Ref(epsilon), d)
    m_ww = mean(ww, dims=2)
    return (w = prod(m_ww), logw = sum(log, m_ww))
end

