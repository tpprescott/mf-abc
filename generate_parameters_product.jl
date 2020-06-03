export ProductGenerator

struct ProductGenerator{Θ, N, T<:NTuple{N, AbstractGenerator}} <: AbstractGenerator{Θ}
    qq::T
    function ProductGenerator(q::Vararg{AbstractGenerator,N}) where N
        T = typeof(q)
        Θ = merge(domain.(q)...)
        return new{Θ, N, T}(q)
    end
end
ProductGenerator(q::AbstractGenerator) = q
function (Q::ProductGenerator{Θ})(; kwargs...) where Θ
    θ, logq = zip(map(q_i -> q_i(; kwargs...), Q.qq)...)
    return merge(θ...), sum(logq)
end
function logpdf(Q::ProductGenerator{Θ}, θ::Θ) where Θ
    return sum(logpdf.(Q.qq, Ref(θ)))
end