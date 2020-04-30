using Statistics
using Distances

export ABCLikelihood
struct ABCLikelihood{F<:AbstractSimulator, D<:Metric, Y<:NamedTuple, SimOptions<:NamedTuple} <: LikelihoodFreeLikelihoodFunction{F}
    f::F
    distance::D
    ϵ::Float64
    y::Array{Y,1}
    kw::SimOptions

    function ABCLikelihood(f::F, distance::D, ϵ::Float64, y::Array{Y,1}; numReplicates::Int64=length(y), kwargs...) where F where D where Y
        kw = merge((numReplicates=numReplicates,), kwargs)
        return new{F, D, eltype(F), typeof(kw)}(f, distance, ϵ, y, kw)
    end
end

function ABCLikelihood(f::F, distance::D, ϵ::Float64; numReplicates::Int64=1, kwargs...) where F where D
    Y = eltype(F)
    y = Array{Y,1}()
    return ABCLikelihood(f, distance, ϵ, y; numReplicates=numReplicates, kwargs...)
end
function ABCLikelihood(L::ABCLikelihood{F, D}, y::Array{Y,1}; kwargs...) where F where D where Y
    return ABCLikelihood(L.f, L.distance, L.ϵ,  y; L.kw..., kwargs...)
end 


function (L::ABCLikelihood)(θ::AbstractModel, args...; kwargs...)
    y = simulate(L.f, θ, args...; L.kw..., kwargs...)
    return ABCLikelihood(L, y)
end

export accept_reject

function (L::ABCLikelihood)(y_obs; loglikelihood::Bool, kwargs...)
    
    t = table(L.y)
    y = select(t, :y) # Requires output of simulation for comparison with data to be called "y"
    isinteger(length(y)//length(y_obs)) || error("Need number of simulations to be an integer multiple of length(y_obs) = $(length(y_obs)).")
    
    y_rs = reshape(y, length(y_obs), :)
    dist = evaluate.(Ref(L.distance), y_obs, y_rs)
    return accept_reject(L.ϵ, dist; loglikelihood=loglikelihood)
end

_ar(epsilon, x) = Float64(x<epsilon)
function accept_reject(epsilon, d; loglikelihood::Bool) 
    ww = _ar.(Ref(epsilon), d)
    m_ww = mean(ww, dims=2)
    return loglikelihood ? sum(log, m_ww) : prod(m_ww)
end