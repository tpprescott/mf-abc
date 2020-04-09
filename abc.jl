module ABC

using ..LikelihoodFree
using Statistics: mean

# ABC gives a specific subtype of AbstractComparison{U,Y} that relies on a distance metric and a threshold value
export AbstractDistance
abstract type AbstractDistance{T} end

export ABCComparison, ABCWeight
struct ABCComparison{T, D<:AbstractDistance{T}} <: AbstractComparison{T}
    d::D
    epsilon::Float64
end
ABCWeight{M,T,TF,TC<:ABCComparison} = LikelihoodFreeWeight{M,T,TF,TC}
ABCWeight(F::TF, C::TC, K::Int64=1) where TF where TC<:ABCComparison = LikelihoodFreeWeight(F, C, K)
ABCWeight(F::TF, d::D, epsilon::Float64, K::Int64=1) where TF where D<:AbstractDistance = ABCWeight(F, ABCComparison(d, epsilon), K)

import .LikelihoodFree.compare, .LikelihoodFree.compare!
export measure, accept_reject, measure!, accept_reject!, compare, compare!

function compare(c::ABCComparison{T}, y::AbstractArray{T,2}, y_obs::AbstractArray{T,1})::NamedTuple where T
    dist = measure(c.d, y, y_obs)
    out = accept_reject(c.epsilon, dist[:d])
    return merge(out, dist)
end
function compare!(ww::AbstractArray{Float64,1}, c::ABCComparison{T}, yy::AbstractArray{T,4}, y_obs)::NamedTuple where T
    dd = Array{Float64,3}(undef, size(yy,2), size(yy,3), size(yy,4))
    save1 = measure!(dd, c.d, yy, y_obs)
    save2 = accept_reject!(ww, c.epsilon, dd)
    return merge((dd=dd,), save1, save2)
end

function measure(d::AbstractDistance{T}, y::AbstractArray{T,2}, y_obs::AbstractArray{T,1})::NamedTuple where T
    dd = Array{Float64,1}(undef, size(y,2))
    for nReplicate in axes(y,2)
        dd[nReplicate] = d(view(y, :, nReplicate), y_obs)
    end
    return (d=dd,)
end
function measure!(dd::AbstractArray{Float64,3}, d::AbstractDistance{T}, yy::AbstractArray{T,4}, y_obs) where T
    size(dd)==(size(yy,2),size(yy,3),size(yy,4)) || error("Mismatched dimensions between distances and simulations")
    size(y_obs,2)==size(dd,3) || error("Mismatched dimensions for number of data points")
    for (nParameter, nReplicate, nDataPoint) in Iterators.product(axes(dd)...)
        dd[nParameter, nReplicate, nDataPoint] = d(view(yy, :, nParameter, nReplicate, nDataPoint), view(y_obs, :, nDataPoint))
    end
    return NamedTuple()
end

_ar(epsilon, x) = Float64(x<epsilon)
function accept_reject(epsilon::Float64, d::AbstractArray{Float64,1})::NamedTuple{(:w,), Tuple{Float64}}
    ww = _ar.(Ref(epsilon), d)
    return (w = mean(ww),)
end
function accept_reject!(ww::AbstractArray{Float64,1}, epsilon::Float64, dd::AbstractArray{Float64, 3})::NamedTuple
    for nParameter in axes(dd,1)
        ww[nParameter] = 1.0
        for nDataPoint in axes(dd, 3)
            if !iszero(ww[nParameter])
                ww[nParameter] *= accept_reject(epsilon, view(dd, nParameter, :, nDataPoint))[:w]
            end
        end
    end
    return NamedTuple()
end


import .LikelihoodFree.rejection_sample
export rejection_sample
function rejection_sample(y_obs, q::AbstractGenerator{M}, F::AbstractSimulator{M,T}, d::AbstractDistance{T}, epsilon::Float64, N) where M where T
    return rejection_sample(y_obs, q, ABCWeight(epsilon, d), N)
end

end

