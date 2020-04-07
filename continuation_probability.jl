# Auxiliary functions
isp(x)::Bool = zero(x) < x <=one(x)
ispos(x)::Bool = zero(x) < x

# Call the continuation probability
function continuation_probability(
    x::H,
    m::M, 
    u::U,
    path..., # Sequence of output from "weight", most recent first
)::Float64 where H<:AbstractContinuationProbability{M,U,Y} where M where U where Y
    x(m, u, path...)
end

# Implement some continuation probabilities
export Go
struct Go{M,U,Y} <: AbstractContinuationProbability{M,U,Y} end
(::Go)(args...) = one(Float64)

export EarlyDecision
struct EarlyDecision{M,U,Y} <: AbstractContinuationProbability{M,U,Y}
    eta::Float64
    function EarlyDecision{M,U,Y}(x) where M where U where Y
        isp(x) ? new{M,U,Y}(x) : error("Not probability")
    end
end
(H::EarlyDecision)(args...) = H.eta

export EarlyAcceptReject

struct EarlyAcceptReject{M, U, Y, C<:AbstractComparison{U,Y}} <: AbstractContinuationProbability{M, U, Y}
    eta1::Float64
    eta2::Float64
    comparison::C
    function EarlyAcceptReject{M}(eta1, eta2, comparison::C) where M where C<:AbstractComparison{U,Y} where U where Y
        (isp(eta1) & isp(eta2)) ? new{M, U, Y, C}(eta1, eta2, comparison) : error("Invalid continuation probabilities") 
    end
end
function EarlyAcceptReject{M}(eta1, eta2, d::D, epsilon::Float64) where M where D<:AbstractDistance{Y} where U where Y
    return EarlyAcceptReject{M}(eta1, eta2, ABCComparison(d, epsilon))
end
function (H::EarlyAcceptReject{M,U,Y})(m::M, u::U, last_node, path...)::Float64 where M where U where Y
    wt = compare(H.comparison, u, last_node[:y])
    return H.eta2 + wt[:w]*(H.eta1 - H.eta2) 
end

export EarlyReject
EarlyReject(x::Float64, args...) = EarlyAcceptReject(1.0, x, args...)


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