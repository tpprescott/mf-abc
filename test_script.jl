using Distributions: Uniform, MvNormal, Normal, product_distribution
using LinearAlgebra: diagm

TomM = NamedTuple{(:x, :σ), Tuple{Float64,Float64}}
q = DistributionGenerator(TomM, product_distribution([Normal(0.5,0.1), Normal(0.5, 0.1)]))
prior = DistributionGenerator(TomM, product_distribution([Uniform(0,1), Uniform(0,1)]))

y_obs = [[0.1, 0.12] .+ 0.2*randn(2) for i in 1:10]

struct TomF <: AbstractSimulator end
function (::TomF)(; x::Float64, σ::Float64, n=randn(2), kwargs...)::NamedTuple 
    y = x .+ σ*n
    return (y=y, n=n)
end
F = TomF()

using Distances

L_abc = ABCLikelihood(F, Euclidean(), 0.1, num_simulations = 10*length(y_obs))
L_sl = BayesianSyntheticLikelihood(F, num_simulations = 1000)

Σ_abc = MonteCarloProposal(prior, q, L_abc, y_obs)
Σ_sl = MonteCarloProposal(prior, q, L_sl, y_obs)

K = PerturbationKernel{TomM}(MvNormal(zeros(2), diagm([0.1,0.1])))
Σ = MonteCarloProposal(prior, K, L_sl, y_obs)