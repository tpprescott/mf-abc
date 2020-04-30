using Distributions: Uniform, MvNormal, Normal, product_distribution
using LinearAlgebra: diagm

TomM = NamedTuple{(:x, :σ), Tuple{Float64,Float64}}
q = DistributionGenerator(TomM, product_distribution([Normal(0.5,0.1), Normal(0.5, 0.1)]))
prior = DistributionGenerator(TomM, product_distribution([Uniform(0,1), Uniform(0,1)]))

y_obs = [[0.1, 0.12] .+ 0.2*randn(2) for i in 1:10]

struct TomF <: AbstractSimulator end
function (::TomF)(; x::Float64, σ::Float64, n=randn(2), kwargs...)
    y = x .+ σ*n
    return (y=y, n=n)
end
F = TomF()
Base.eltype(::Type{TomF}) = NamedTuple{(:y, :n), Tuple{Array{Float64,1}, Array{Float64,1}}}

using Distances

L_abc = ABCLikelihood(F, Euclidean(), 0.1, numReplicates=100)
L_bsl = BayesianSyntheticLikelihood(F, numReplicates=500)
L_csl = BayesianSyntheticLikelihood(F, numReplicates=500, numIndependent=10)

C0 = diagm(0=>[0.02,0.02])
Σ =  MCMCProposal(prior, C0, L_bsl, y_obs)
Σc = MCMCProposal(prior, C0, L_csl, y_obs)