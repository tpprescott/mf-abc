module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra

const EMF = ConstantEF(1.0)

# NoEF
const prior_NoEF = DistributionGenerator(SingleCellModel_NoEF, product_distribution([
    Uniform(0,2), Uniform(0,5), Uniform(0,5), Uniform(0,2)
]))
const K_NoEF = PerturbationKernel{SingleCellModel_NoEF}(
    MvNormal(zeros(4), diagm(0=>fill(0.01, 4)))
)
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)
L_NoEF_BSL(s, n) = BayesianSyntheticLikelihood(F_NoEF, s, numReplicates=n)
L_NoEF_CSL(s, n, i) = BayesianSyntheticLikelihood(F_NoEF, s, numReplicates=n, numIndependent=i)

export smc_generations
smc_generations(T::Int64, N::Int64) = smc_generations(T-1:-1:0, fill(N,T))
smc_generations(s::AbstractArray{Int64,1}, n::AbstractArray{Int64,1}) = smc_generations(zip(s, n)...)
smc_generations(tup::Tuple) = (((L_NoEF_BSL(tup...), ), (y_obs_NoEF,)),)
smc_generations(tup::Tuple, tups::Tuple...) = (smc_generations(tup)..., smc_generations(tups...)...)


export Σ_NoEF_BSL_MC, Σ_NoEF_BSL_IS, Σ_NoEF_BSL_SMC, Σ_NoEF_CSL_MC

Σ_NoEF_BSL_MC(s, n) = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_BSL(s, n), y_obs_NoEF)
Σ_NoEF_BSL_IS(s, n) = ISProposal(prior_NoEF, L_NoEF_BSL(s, n), y_obs_NoEF)
Σ_NoEF_BSL_SMC(s, n) = SMCWrapper(prior_NoEF, smc_generations(s, n))
Σ_NoEF_CSL_MC(s,n,i) = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_CSL(s, n, i), y_obs_NoEF)

# EF
const EMF = ConstantEF(1.0)
const prior_EF = DistributionGenerator(SingleCellModel_EF, product_distribution([
    Uniform(0,2), Uniform(0,5), Uniform(0,5), Uniform(0,2), Uniform(0,2)
]))
const K_EF = PerturbationKernel{SingleCellModel_EF}(
    MvNormal(zeros(5), diagm(0=>fill(0.01, 5)))
)
const y_obs_EF = vcat.(EF_displacements, EF_angles)
const F_EF = SingleCellSimulator(σ_init=0.1, angles=true, emf = EMF)
const L_EF_BSL = BayesianSyntheticLikelihood(F_EF, numReplicates=50)
const L_EF_CSL = BayesianSyntheticLikelihood(F_EF, numReplicates=50, numIndependent=1)

export Σ_EF_BSL_MC, Σ_EF_BSL_IS, Σ_EF_CSL_MC
const Σ_EF_BSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_BSL, y_obs_EF)
const Σ_EF_BSL_IS = ISProposal(prior_EF, L_EF_BSL, y_obs_EF)
const Σ_EF_CSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_CSL, y_obs_EF)

# Joint
export Σ_Joint_BSL_MC, Σ_Joint_BSL_IS, Σ_Joint_CSL_MC
const Σ_Joint_BSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_BSL(0, 50), L_EF_BSL), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_IS = ISProposal(prior_EF, (L_NoEF_BSL(0, 50), L_EF_BSL), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_CSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_CSL(0, 50,1), L_EF_CSL), (y_obs_NoEF, y_obs_EF))

end