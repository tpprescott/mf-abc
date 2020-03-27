module ABC

using ..LikelihoodFree

# ABC gives a specific subtype of AbstractComparison{U,Y} that relies on a distance metric and a threshold value
export AbstractDistance
abstract type AbstractDistance{U,Y} end

export ABCComparison, ABCWeight
struct ABCComparison{U, Y, D<:AbstractDistance{U,Y}} <: AbstractComparison{U,Y}
    epsilon::Float64
    d::D
end
ABCWeight{M,U,Y,TF,TC<:ABCComparison} = LikelihoodFreeWeight{M,U,Y,TF,TC}

import .LikelihoodFree.compare, .LikelihoodFree.compare!
export measure, accept_reject, compare

function measure(d::AbstractDistance{U, Y}, u::U, y::Y)::Float64 where U where Y
    return d(u, y)
end
function measure!(d::AbstractDistance{U,Y}, u::U; yy::AbstractArray{Y}, dd::AbstractArray{Float64}, saved...)::Nothing where U where Y
    for i in eachindex(yy)
        dd[i] = measure(d, u, yy[i])
    end
    return nothing
end
function measure(d::AbstractDistance{U,Y}, u::U; yy::AbstractArray{Y}, saved...)::Array{Float64} where U where Y
    dd = Array{Float64}(undef, size(yy))
    measure!(d, u; yy=yy, dd=dd, saved...)
    return dd
end

function accept_reject(epsilon::Float64, d::Float64)::Float64
    return d<epsilon ? one(Float64) : zero(Float64) 
end
function accept_reject!(ar::AbstractArray{Float64}, epsilon::Float64; dd::AbstractArray{Float64}, saved...)::Nothing
    for i in eachindex(dd)
        ar[i] = accept_reject(epsilon, dd[i])
    end
end
function accept_reject(epsilon::Float64; dd::AbstractArray{Float64}, saved...)::AbstractArray{Float64}
    ar = Array{Float64}(undef, size(dd))
    accept_reject!(ar, epsilon; dd=dd, saved...)
    return ar
end

function compare(c::ABCComparison{U,Y}, u::U, y::Y)::Float64 where U where Y
    d = measure(c.d, u, y)
    out = accept_reject(c.epsilon, d)
    return Float64(out)
end
function compare!(
    out::Array{Float64}, 
    c::ABCComparison{U,Y}, 
    u::U; 
    yy::AbstractArray{Y}, 
    dd::AbstractArray{Float64}=Array{Float64}(undef, size(out)),
    saved...
)::Nothing where U where Y

    measure!(c.d, u; yy=yy, dd=dd, saved...)
    accept_reject!(out, c.epsilon; dd=dd, saved...)
    return nothing
end
function compare(c::ABCComparison{U,Y}, u::U; yy::AbstractArray{Y}, saved...)::AbstractArray{Float64} where U where Y
    out = Array{Float64}(undef, size(yy))
    dd = Array{Float64}(undef, size(yy))
    compare!(out, c, u; yy=yy, dd=dd, saved...)
    return out
end

import .LikelihoodFree.sample
export sample
function sample(u::U, q::AbstractGenerator{M}, F::AbstractSimulator{M,U,Y}, d::AbstractDistance{U,Y}, epsilon::Float64, N; saved...) where M where U where Y
    dd = Array{Float64}(undef, N)
    return sample(u, q, F, ABCComparison(epsilon, d), N; dd=dd, saved...)
end

end

