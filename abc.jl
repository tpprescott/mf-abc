module ABC

using ..LikelihoodFree
using Statistics: mean

# ABC gives a specific subtype of AbstractComparison{U,Y} that relies on a distance metric and a threshold value
export AbstractDistance
abstract type AbstractDistance{U,Y} end

export ABCComparison, ABCWeight
struct ABCComparison{U, Y, D<:AbstractDistance{U,Y}} <: AbstractComparison{U,Y}
    d::D
    epsilon::Float64
end
ABCWeight{M,U,Y,TF,TC<:ABCComparison} = LikelihoodFreeWeight{M,U,Y,TF,TC}
ABCWeight(F::TF, C::TC, K::Int64=1) where TF where TC<:ABCComparison = LikelihoodFreeWeight(F, C, K)
ABCWeight(F::TF, d::D, epsilon::Float64, K::Int64=1) where TF where D<:AbstractDistance = ABCWeight(F, ABCComparison(d, epsilon), K)

import .LikelihoodFree.compare, .LikelihoodFree.compare!
export measure, accept_reject, compare

function measure(d::AbstractDistance{U, Y}, u::U, y::Y)::NamedTuple where U where Y
    return (d = d(u, y),)
end
function measure(d::AbstractDistance{U, Y}, u::U, y::AbstractArray{Y,1}) where U where Y
    d_i = Array{Float64,1}(undef, length(y))
    save = measure!(d_i, d, u, y)
    return merge((d = d_i,), save)
end
function measure!(d_i::AbstractArray{Float64,1}, d::AbstractDistance{U, Y}, u::U, y::AbstractArray{Y,1}) where U where Y
    for i in eachindex(d_i)
        d_i[i] = measure(d, u, y[i])[:d]
    end
    return NamedTuple()
end
function measure(d::AbstractDistance{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    dd = Array{Float64,2}(undef, size(yy))
    save = measure!(dd, d, u, yy)
    return merge((dd=dd,),save)
end
function measure!(dd::AbstractArray{Float64,2}, d::AbstractDistance{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    for i in eachindex(yy)
        dd[i] = measure(d, u, yy[i])[:d]
    end
    return NamedTuple()
end

function accept_reject(epsilon::Float64, d::Float64)::NamedTuple
    ar = d<epsilon ? one(Float64) : zero(Float64)
    return (w = ar,)
end
function accept_reject(epsilon::Float64, d::AbstractArray{Float64,1})::NamedTuple
    f(x) = accept_reject(epsilon, x)[:w]
    ar = mean(f, d)
    return (w = ar,)
end
function accept_reject(epsilon::Float64, dd::AbstractArray{Float64,2})::NamedTuple
    ww = Array{Float64,1}(undef, size(dd,1))
    save = accept_reject!(ww, epsilon, dd)
    return merge((ww=ww,), save)
end
function accept_reject!(ww::AbstractArray{Float64,1}, epsilon::Float64, dd::AbstractArray{Float64,2})::NamedTuple
    for i in eachindex(ww)
        ww[i] = accept_reject(epsilon, view(dd,i,:))[:w]
    end
    return NamedTuple()
end

function compare(c::ABCComparison{U,Y}, u::U, y::AbstractArray{Y,1})::NamedTuple where U where Y
    dist = measure(c.d, u, y)
    out = accept_reject(c.epsilon, dist[:d])
    return merge(out, dist)
end
function compare(c::ABCComparison{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    out = Array{Float64,1}(undef, size(yy,1))
    save = compare!(out, c, u, yy)
    return merge((ww = out,), save)
end
function compare!(out::Array{Float64,1}, c::ABCComparison{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    dd = Array{Float64,2}(undef, size(yy))
    save1 = measure!(dd, c.d, u, yy)
    save2 = accept_reject!(out, c.epsilon, dd)
    return merge((dd=dd,), save1, save2)
end

import .LikelihoodFree.rejection_sample
export rejection_sample
function rejection_sample(u::U, q::AbstractGenerator{M}, F::AbstractSimulator{M,U,Y}, d::AbstractDistance{U,Y}, epsilon::Float64, N) where M where U where Y
    return rejection_sample(u, q, ABCWeight(epsilon, d), N)
end

end

