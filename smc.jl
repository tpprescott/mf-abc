# SMC Sampling

export SMCWrapper

mutable struct SMCWrapper{N, M, Θ, Π<:AbstractGenerator{Θ}, W<:NTuple{N, LikelihoodObservationSet}}
    prior::Π
    lh_set::W
    numSimulations::NTuple{N, Int64}
    scale::NTuple{N, Float64}
    test_idx::NTuple{M, Array{Int64,1}}
    function SMCWrapper(prior::Π, lh_set::W) where W where Π<:AbstractGenerator{Θ} where Θ
        N = length(lh_set)
        M = length(lh_set[end][1])
        Σ = new{N, M, Θ, Π, W}()
        Σ.prior = prior
        Σ.lh_set = lh_set
        return Σ
    end
end

SequentialImportanceDistribution(
    tn::IndexedTable,
    prior::AbstractGenerator,
    delta::Float64=0.0,
) = SequentialImportanceDistribution(select(tn, :θ), select(tn, :weight), prior, delta)

function Base.iterate(
    Σ::SMCWrapper{N, M, Θ},
    (n, qn)::Tuple{Int64, AbstractGenerator{Θ}} = (0, Σ.prior)
) where N where M where Θ

    if n>=N
        return nothing
    else
        n += 1
        println("Generation $n of $N")
        
        Σn = ISProposal(Σ.prior, qn, Σ.lh_set[n])
        tn = importance_sample(Σn, Σ.numSimulations[n], scale=Σ.scale[n], test_idx=Σ.test_idx)
        qn1 = n==N ? qn : SequentialImportanceDistribution(tn, Σ.prior)
        return tn, (n, qn1)
    end
end

Base.length(::SMCWrapper{N}) where N = N
Base.eltype(::Type{SMC}) where SMC<:SMCWrapper = IndexedTable

export smc_sample
function smc_sample(
    Σ::SMCWrapper{N, M},
    numSimulations::NTuple{N,Int64};
    scale::NTuple{N,Number} = Tuple(Iterators.repeated(1, N)),
    test_idx::NTuple{M,AbstractArray{Int64,1}} = Tuple(Iterators.repeated(Array{Int64,1}(), M))
    ) where N where M

    Σ.numSimulations = numSimulations
    Σ.scale = scale
    Σ.test_idx = test_idx
    return collect(Σ)
end