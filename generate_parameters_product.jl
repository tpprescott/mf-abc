struct ProductGenerator{Θ, N, T<:NTuple{N, AbstractGenerator}} <: AbstractGenerator{Θ}
    qq::T
    function ProductGenerator(q::Vararg{AbstractGenerator,N}) where N
        T = typeof(q)
        Θ = merge(domain.(q)...)
        return new{Θ, N, T}(q)
    end
end
function (Q::ProductGenerator{Θ})(; prior::AbstractGenerator{Θ}, kwargs...) where Θ
    iter = map(q_i -> q_i(; prior=q_i, kwargs...), Q.qq)
    θ = merge((r.θ for r in iter)...)
    logq = sum(r.logq for r in iter)
    logp = prior==Q ? logq : logpdf(prior, θ)
    return (θ=θ , logq=logq, logp=logp)
end
function logpdf(Q::ProductGenerator{Θ}, θ::Θ) where Θ
    return sum(logpdf.(Q.qq, Ref(θ)))
end