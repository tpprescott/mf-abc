export AbstractMFABCDistance
abstract type AbstractMFABCDistance{TY, U, Y<:MFABCOutput{TY}} <: AbstractDistance{U, Y} end

export MFABCWeight, MFABCDistance
struct MFABCDistance{TY, U, Y} <: AbstractMFABCDistance{TY, U, Y}
    function MFABCDistance{Y}(u::U) where U<:Experiment where Y<:MFABCOutput{TY} where TY
        return new{TY, U, Y}()
    end
    function MFABCDistance{U,Y}() where U where Y<:MFABCOutput{TY} where TY
        return new{TY, U, Y}()
    end
end
# TODO: Other concrete distances that may depend on simulation output (i.e. adaptive epsilon, TSFresh, etc)

import .ABC.measure_distance
function measure_distance(d::AbstractMFABCDistance{TY, U, Y}, u::U, y::Y_i)::Float64  where Y_i<:TY where TY where U where Y
    return d(u, y)
end
# Need to specify (d::D{TY,U,Y})(u::U, y::Y_i)::Float64 for all Y_i<:TY of interest, for each concrete type of distance D
# Example:
# function (d::MFABCDistance{TY, U, Y}(u::U, y_i::Y_i)) where Y_i<:TY where TY
# return abs(u.y_obs - y_i.y)
# end

struct MFABCWeight{U, Y, TY, D<:AbstractMFABCDistance{TY, U, Y}} <: Weight{U, Y}
    eps::Vector{Float64}
    dist::D
    function MFABCWeight(x::Vector{Float64}, d::D) where D <: AbstractMFABCDistance{TY,U,Y} where TY where U where Y
        all(ispos, x) || error("Non-positive threshold(s)")
        return new{U,Y,TY,D}(x,d)
    end
end

function (w::MFABCWeight{U, Y, TY, D})(u::U, out::Y)::Float64 where U where Y where TY where D
    wt = zero(Float64)
    for (v, y_v, eta_v) in zip(out.seq, out.y, out.eta)
        d::Float64 = measure_distance(w.dist, u, y_v)
        corr = Float64(d < w.eps[v]) - wt
        wt += corr/eta_v
    end
    return wt
end