struct ProductGenerator{Θ, N, T<:NTuple{N, AbstractGenerator}} <: AbstractGenerator{Θ}
    qq::T
    function ProductGenerator(q::Vararg{AbstractGenerator,N}) where N
        T = typeof(q)
        Θ = merge(domain.(q)...)
        return new{Θ, N, T}(q)
    end
end
function (Q::ProductGenerator{Θ})(; kwargs...) where Θ
    iter = map(q_i -> q_i(; kwargs...), Q.qq)
    θ = merge((r.θ for r in iter)...)
    logq = sum(r.logq for r in iter)
    return θ, logq
end
function logpdf(Q::ProductGenerator{Θ}, θ::Θ) where Θ
    return sum(logpdf.(Q.qq, Ref(θ)))
end