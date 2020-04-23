using Distributions: Uniform, MvNormal, Normal, product_distribution

TomM = NamedTuple{(:x, :σ), Tuple{Float64,Float64}}
q = DistributionGenerator(TomM, product_distribution([Normal(0.5,0.1), Normal(0.5, 0.1)]))
prior = DistributionGenerator(TomM, product_distribution([Uniform(0,1), Uniform(0,1)]))

y_obs = [[0.1, 0.12] .+ 0.2*randn(2) for i in 1:10]

struct TomF <: AbstractSimulator end
function (::TomF)(K::Int64; x::Float64, σ::Float64, n=randn(2), kwargs...)::NamedTuple 
    y = x .+ σ*n
    return (y=y, n=n)
end
F = TomF()

using Distances
c_abc = ABCComparison(Euclidean(), 0.5)
w_abc = LikelihoodFreeWeight(F, c_abc, 10)

c_sl = SyntheticLikelihood()
w_sl = LikelihoodFreeWeight(F, c_sl, 50)

Σ_abc = MonteCarloProposal(prior, q, w_abc, y_obs)
Σ_sl = MonteCarloProposal(prior, q, w_sl, y_obs)

K = PerturbationKernel{TomM}(MvNormal(zeros(2), LinearAlgebra.diagm([0.1,0.1])))
Σ = MonteCarloProposal(prior, K, w_sl, y_obs)