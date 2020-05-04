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
    MvNormal(zeros(4), diagm(0=>fill(0.1, 4)))
)
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)
const L_NoEF_BSL = BayesianSyntheticLikelihood(F_NoEF, numReplicates=500)
const L_NoEF_CSL = BayesianSyntheticLikelihood(F_NoEF, numReplicates=500, numIndependent=5)

const Σ_NoEF_BSL = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_BSL, y_obs_NoEF)
const Σ_NoEF_CSL = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_CSL, y_obs_NoEF)


# EF
const EMF = ConstantEF(1.0)
const prior_EF = DistributionGenerator(SingleCellModel_EF, product_distribution([
    Uniform(0,2), Uniform(0,5), Uniform(0,5), Uniform(0,2), Uniform(0,2)
]))
const K_EF = PerturbationKernel{SingleCellModel_EF}(
    MvNormal(zeros(5), diagm(0=>fill(0.1, 5)))
)
const y_obs_EF = vcat.(EF_displacements, EF_angles)
const F_EF = SingleCellSimulator(σ_init=0.1, angles=true, emf = EMF)
const L_EF_BSL = BayesianSyntheticLikelihood(F_EF, numReplicates=500)
const L_EF_CSL = BayesianSyntheticLikelihood(F_EF, numReplicates=500, numIndependent=5)

const Σ_EF_BSL = MCMCProposal(prior_EF, K_EF, L_EF_BSL, y_obs_EF)
const Σ_EF_CSL = MCMCProposal(prior_EF, K_EF, L_EF_CSL, y_obs_EF)

# Joint
const Σ_Joint_CSL = MCMCProposal(prior_EF, K_EF, (L_NoEF_CSL, L_EF_CSL), (y_obs_NoEF, y_obs_EF))

export MCMC_NoEF_BSL, MCMC_NoEF_CSL
MCMC_NoEF_BSL(N) = Iterators.take(Σ_NoEF_BSL, N)
MCMC_NoEF_CSL(N) = Iterators.take(Σ_NoEF_CSL, N)

export MCMC_EF_BSL, MCMC_EF_CSL
MCMC_EF_BSL(N) = Iterators.take(Σ_EF_BSL, N)
MCMC_EF_CSL(N) = Iterators.take(Σ_EF_CSL, N)

export MCMC_Joint_CSL
MCMC_Joint_CSL(N) = Iterators.take(Σ_EF_BSL, N)

end