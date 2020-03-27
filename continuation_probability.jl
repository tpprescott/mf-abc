export AbstractContinuationProbability

# Auxiliary functions
isp(x)::Bool = zero(x) < x <=one(x)
ispos(x)::Bool = zero(x) < x

# Call the continuation probability
function continuation_probability(x::AbstractContinuationProbability, m, u, output)::Float64
    x(m, u, output)
end

# Implement some continuation probabilities
export Go
struct Go{U,Y} <: AbstractContinuationProbability{U,Y} end
(::Go)(args...) = one(Float64)

export EarlyDecision
struct EarlyDecision{U,Y} <: AbstractContinuationProbability{U,Y}
    eta::Float64
    function EarlyDecision{U,Y}(x) where U where Y
        isp(x) ? new(x) : error("Not probability")
    end
end
(H::EarlyDecision)(args...) = H.eta

export EarlyAcceptReject

struct EarlyAcceptReject{U, Y, W<:ABCWeight{U,Y}} <: AbstractContinuationProbability{U,Y}
    eta1::Float64
    eta2::Float64
    wt::W
    function EarlyAcceptReject(eta1, eta2, w::W) where W<:ABCWeight{U,Y} where U where Y
        (isp(eta1) & isp(eta2)) ? new{U, Y, W}(eta1, eta2, w) : error("Invalid continuation probabilities") 
    end
end
function EarlyAcceptReject(eta1, eta2, d::ABCDistance{U,Y}, epsilon::Float64) where U where Y
    return EarlyAcceptReject(eta1, eta2, ABCWeight(epsilon, d))
end
function (H::EarlyAcceptReject{U,Y})(m, u, output)::Float64 where U where Y
    return H.eta2 + H.wt(u, output.y[end])*(H.eta1 - H.eta2) 
end

export EarlyReject
EarlyReject(x::Float64, args...) = EarlyAcceptReject(1.0, x, args...)

struct ZeroDistance{U,Y} <: AbstractABCDistance{U,Y} end
function (d::ZeroDistance{U,Y})(u::U, y::Y) where U where Y
    return zero(Float64)
end

# Might be possible to play with promotions and conversions to speed things up
#=
import Base.convert
import Base.promote_rule
convert(::Type{EarlyDecision{U,Y}}, x::Go{U,Y}) where U where Y = EarlyDecision{U,Y}(1.0)
convert(::Type{EarlyAcceptReject{U,Y,W}}, x::Go{U,Y}) where U where Y where W = EarlyAcceptReject(1.0, 1.0, ABCWeight(Inf, ZeroDistance{U,Y}()))
convert(::Type{EarlyAcceptReject{U,Y,W}}, x::EarlyDecision{U,Y}) where U where Y = EarlyAcceptReject(x.eta, x.eta, ABCWeight(Inf, ZeroDistance{U,Y}()))
promote_rule(::Type{Go{U,Y}}, ::Type{EarlyAcceptReject{U,Y,W}}) where U where Y where W = EarlyAcceptReject{U,Y,W}
promote_rule(::Type{EarlyDecision{U,Y}}, ::Type{EarlyAcceptReject{U,Y,W}}) where U where Y where W = EarlyAcceptReject{U,Y,W}
promote_rule(::Type{Go{U,Y}}, ::Type{EarlyDecision{U,Y}}) where U where Y = EarlyDecision{U,Y}
=#