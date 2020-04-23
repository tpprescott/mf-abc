module ABC

using ..LikelihoodFree
using Statistics: mean
using Distances


export ABCComparison, ABCWeight
struct ABCComparison{D<:Metric} <: AbstractComparison
    d::D
    epsilon::Float64
end

ABCWeight{Simulator, Comparison<:ABCComparison} = LikelihoodFreeWeight{Simulator, Comparison}
ABCWeight(F::Simulator, C::Comparison, K::Int64=1) where Simulator where Comparison<:ABCComparison = LikelihoodFreeWeight(F, C, K)
ABCWeight(F::Simulator, d::D, epsilon::Float64, K::Int64=1) where Simulator where D<:Metric = ABCWeight(F, ABCComparison(d, epsilon), K)

export accept_reject

function (C::ABCComparison)(y_obs::Array{Array{T,1},1}, y::Array{Array{T,1},1}; kwargs...) where T
    isinteger(length(y)//length(y_obs)) || error("Need number of simulations to be an integer multiple of length(y_obs) = $(length(y_obs)).")
    
    y_rs = reshape(y, length(y_obs), :)
    dist = evaluate.(Ref(C.d), y_obs, y_rs)
    compare_out::NamedTuple = accept_reject(C.epsilon, dist)
    return merge(compare_out, (d=vec(dist),))
end

_ar(epsilon, x) = Float64(x<epsilon)
function accept_reject(epsilon, d) where T 
    ww = _ar.(Ref(epsilon), d)
    m_ww = mean(ww, dims=2)
    return (w = prod(m_ww), logw = sum(log, m_ww))
end

end

