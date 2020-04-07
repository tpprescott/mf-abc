module ABC

using ..LikelihoodFree

# ABC gives a specific subtype of AbstractComparison{U,Y} that relies on a distance metric and a threshold value
export AbstractDistance
abstract type AbstractDistance{Y <: AbstractSummaryStatisticSpace} end

export ABCComparison, ABCWeight
struct ABCComparison{U, Y, D<:AbstractDistance{Y}} <: AbstractComparison{U,Y}
    d::D
    epsilon::Float64
end
ABCWeight{M,U,Y,TF,TC<:ABCComparison} = LikelihoodFreeWeight{M,U,Y,TF,TC}
ABCWeight(F::TF, C::TC) where TF where TC<:ABCComparison = LikelihoodFreeWeight(F, C)
ABCWeight(F::TF, d::D, epsilon::Float64) where TF where D<:AbstractDistance = ABCWeight(F, ABCComparison(d, epsilon))

import .LikelihoodFree.compare, .LikelihoodFree.compare!
export measure, accept_reject, compare

function measure(d::AbstractDistance{Y}, y_obs::Y, y::Y)::NamedTuple where Y
    return (d=d(y_obs, y),)
end
function measure(d::AbstractDistance{Y}, u::U, y::Y)::NamedTuple where U<:AbstractExperiment{Y} where Y
    return measure(d, u.y_obs, y)
end
function measure!(dd::AbstractArray{Float64}, d::AbstractDistance{Y}, u::U, yy::AbstractArray{Y})::NamedTuple where U where Y
    for i in eachindex(yy)
        dd[i] = measure(d, u, yy[i])[:d]
    end
    return NamedTuple()
end
function measure(d::AbstractDistance{Y}, u::U, yy::AbstractArray{Y})::NamedTuple where U where Y
    dd = Array{Float64}(undef, size(yy))
    save = measure!(dd, d, u, yy)
    return merge((dd=dd,),save)
end

function accept_reject(epsilon::Float64, d::Float64)::NamedTuple
    ar = d<epsilon ? one(Float64) : zero(Float64)
    return (w = ar,)
end
function accept_reject!(ww::AbstractArray{Float64}, epsilon::Float64, dd::AbstractArray{Float64})::NamedTuple
    for i in eachindex(dd)
        ww[i] = accept_reject(epsilon, dd[i])[:w]
    end
    return NamedTuple()
end
function accept_reject(epsilon::Float64, dd::AbstractArray{Float64})::NamedTuple
    ww = Array{Float64}(undef, size(dd))
    save = accept_reject!(ww, epsilon, dd)
    return merge((ww=ww,),save)
end

function compare(c::ABCComparison{U,Y}, u::U, y::Y)::NamedTuple where U where Y
    dist = measure(c.d, u, y)
    out = accept_reject(c.epsilon, dist[:d])
    return merge(out, dist)
end
function compare!(
    out::Array{Float64}, 
    c::ABCComparison{U,Y}, 
    u::U,
    yy::AbstractArray{Y}, 
)::NamedTuple where U where Y

    dd = Array{Float64}(undef, size(yy))
    save1 = measure!(dd, c.d, u, yy)
    save2 = accept_reject!(out, c.epsilon, dd)
    return merge((dd=dd,), save1, save2)
end
function compare(c::ABCComparison{U,Y}, u::U, yy::AbstractArray{Y})::NamedTuple where U where Y
    out = Array{Float64}(undef, size(yy))
    save = compare!(out, c, u, yy)
    return merge((ww = out,), save)
end

import .LikelihoodFree.rejection_sample
export rejection_sample
function rejection_sample(u::U, q::AbstractGenerator{M}, F::AbstractSimulator{M,U,Y}, d::AbstractDistance{Y}, epsilon::Float64, N) where M where U where Y
    return rejection_sample(u, q, ABCWeight(epsilon, d), N)
end

end

