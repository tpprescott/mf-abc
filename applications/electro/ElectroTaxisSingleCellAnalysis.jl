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

export Σ_NoEF_BSL_MC, Σ_NoEF_BSL_IS, Σ_NoEF_CSL_MC, Σ_NoEF_BSL_SMC

const Σ_NoEF_BSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_BSL(0, 250), y_obs_NoEF)
const Σ_NoEF_BSL_IS = ISProposal(prior_NoEF, L_NoEF_BSL(0, 250), y_obs_NoEF)
const Σ_NoEF_BSL_SMC = SMCWrapper(prior_NoEF, (
    ((L_NoEF_BSL(4, 50),), (y_obs_NoEF,)),
    ((L_NoEF_BSL(3, 100),), (y_obs_NoEF,)),
    ((L_NoEF_BSL(2, 150),), (y_obs_NoEF,)),
    ((L_NoEF_BSL(1, 200),), (y_obs_NoEF,)),
    ((L_NoEF_BSL(0, 250),), (y_obs_NoEF,)),
),
(500, 500, 500, 500, 500))

Σ_NoEF_CSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_CSL(0, 250, 1), y_obs_NoEF)

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