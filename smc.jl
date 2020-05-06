# SMC Sampling

export SMCWrapper

struct SMCWrapper{N, Θ, Π<:AbstractGenerator{Θ}, W<:NTuple{N, LikelihoodObservationSet}}
    prior::Π
    lh_set::W
    N::NTuple{N, Int64}
end

export importance_weight
importance_weight(t) = exp.(select(t, :logw) .+ select(t, :logp) .- select(t, :logq))

function Base.iterate(
    Σ::SMCWrapper{N, Θ},
    (n, qn)::Tuple{Int64, AbstractGenerator{Θ}} = (0, Σ.prior)
) where N where Θ

    if n>=N
        return nothing
    else
        n += 1
        println("Generation $n of $N")
        
        Σn = ISProposal(Σ.prior, qn, Σ.lh_set[n])
        tn = importance_sample(Σn, Σ.N[n])
        qn1 = SequentialImportanceDistribution(select(tn, :θ), importance_weight(tn), Σ.prior)
        return tn, (n, qn1)
    end
end

Base.length(::SMCWrapper{N}) where N = N
Base.eltype(::Type{SMCWrapper}) = IndexedTable
