using Distributed
@everywhere include("likelihood_free.jl")
@everywhere include("applications/electro/ElectroTaxis.jl")
using .LikelihoodFree
using .ElectroTaxis
using LinearAlgebra, Distances, Distributions, DifferentialEquations, IndexedTables, Plots

prior = DistributionGenerator(SingleCellModel_NoEF, product_distribution([
    Uniform(0,2), Uniform(0,5), Uniform(0,5), Uniform(0,2)
]))
K = PerturbationKernel{SingleCellModel_NoEF}(MvNormal(zeros(4), diagm(0=>fill(0.1, 4))))
F = SingleCellSimulator(σ_init=0.1)
L_abc = ABCLikelihood(F, Euclidean(), 1.0, num_simulations=1*50) # integer multiple of 50
L_bsl = BayesianSyntheticLikelihood(F, num_simulations=500) # ideally >500, although can be slow

Σ_is = MonteCarloProposal(prior, L_bsl, NoEF_displacements)
Σ_mcmc = MonteCarloProposal(prior, K, L_bsl, NoEF_displacements)

# importance_sample(Σ_is, length, 10; loglikelihood=false)
# mcmc_sample(Σ_mcmc, length, 10)