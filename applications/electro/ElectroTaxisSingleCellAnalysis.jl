module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra

const EMF = ConstantEF(1.0)

# NoEF
const prior_NoEF = DistributionGenerator(SingleCellModel_NoEF, product_distribution([
    Uniform(0,3), Uniform(0,5), Uniform(0,5), Uniform(0,2),
]))
const K_NoEF = PerturbationKernel{SingleCellModel_NoEF}(
    MvNormal(zeros(4), diagm(0=>fill(0.01, 4)))
)
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)
L_NoEF_BSL(s, n) = BayesianSyntheticLikelihood(F_NoEF, s, numReplicates=n)
L_NoEF_CSL(s, n, i) = BayesianSyntheticLikelihood(F_NoEF, s, numReplicates=n, numIndependent=i)


export Σ_NoEF_BSL_MC, Σ_NoEF_BSL_IS, Σ_NoEF_BSL_SMC, Σ_NoEF_CSL_MC
const Σ_NoEF_BSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_BSL(0, 500), y_obs_NoEF)
const Σ_NoEF_BSL_IS = ISProposal(prior_NoEF, L_NoEF_BSL(0, 500), y_obs_NoEF)
const Σ_NoEF_BSL_SMC = SMCWrapper(
    prior_NoEF,
    (
        ((L_NoEF_BSL(9,  50),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(8, 100),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(7, 150),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(6, 200),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(5, 250),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(4, 300),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(3, 350),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(2, 400),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(1, 450),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(0, 500),), (y_obs_NoEF,)),
    )
)
const Σ_NoEF_CSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_CSL(0, 500, 5), y_obs_NoEF)


# EF
const EMF = ConstantEF(1.0)
const prior_EF = DistributionGenerator(SingleCellModel_EF, product_distribution([
    Uniform(0,3), Uniform(0,5), Uniform(0,5), Uniform(0,2), Uniform(0,2), Uniform(0,1), Uniform(0,1),
]))
const K_EF = PerturbationKernel{SingleCellModel_EF}(
    MvNormal(zeros(7), diagm(0=>fill(0.01, 7)))
)
const y_obs_EF = vcat.(EF_displacements, EF_angles)
const F_EF = SingleCellSimulator(σ_init=0.1, angles=true, emf = EMF)
L_EF_BSL(s,n) = BayesianSyntheticLikelihood(F_EF, s, numReplicates=n)
L_EF_CSL(s,n,i) = BayesianSyntheticLikelihood(F_EF, s, numReplicates=n, numIndependent=i)

export Σ_EF_BSL_MC, Σ_EF_BSL_IS, Σ_EF_BSL_SMC, Σ_EF_CSL_MC
const Σ_EF_BSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_BSL(0, 500), y_obs_EF)
const Σ_EF_BSL_IS = ISProposal(prior_EF, L_EF_BSL(0, 500), y_obs_EF)
const Σ_EF_BSL_SMC = SMCWrapper(
    prior_EF,
    (
        ((L_EF_BSL(9,  50),), (y_obs_EF,)),
        ((L_EF_BSL(8, 100),), (y_obs_EF,)),
        ((L_EF_BSL(7, 150),), (y_obs_EF,)),
        ((L_EF_BSL(6, 200),), (y_obs_EF,)),
        ((L_EF_BSL(5, 250),), (y_obs_EF,)),
        ((L_EF_BSL(4, 300),), (y_obs_EF,)),
        ((L_EF_BSL(3, 350),), (y_obs_EF,)),
        ((L_EF_BSL(2, 400),), (y_obs_EF,)),
        ((L_EF_BSL(1, 450),), (y_obs_EF,)),
        ((L_EF_BSL(0, 500),), (y_obs_EF,)),
    )
)
const Σ_EF_CSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_CSL(0, 500, 5), y_obs_EF)

# Joint
export Σ_Joint_BSL_MC, Σ_Joint_BSL_IS, Σ_Joint_CSL_MC, Σ_Joint_BSL_SMC
const Σ_Joint_BSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_BSL(0, 500), L_EF_BSL(0, 500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_IS = ISProposal(prior_EF, (L_NoEF_BSL(0, 500), L_EF_BSL(0, 500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_SMC = SMCWrapper(
    prior_EF,
    (
        ((L_NoEF_BSL(9,  50), L_EF_BSL(9,  50)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(8, 100), L_EF_BSL(8, 100)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(7, 150), L_EF_BSL(7, 150)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(6, 200), L_EF_BSL(6, 200)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(5, 250), L_EF_BSL(5, 250)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(4, 300), L_EF_BSL(4, 300)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(3, 350), L_EF_BSL(3, 350)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(2, 400), L_EF_BSL(2, 400)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(1, 450), L_EF_BSL(1, 450)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(0, 500), L_EF_BSL(0, 500)), (y_obs_NoEF, y_obs_EF)),
    )
)
const Σ_Joint_CSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_CSL(0, 500 , 5), L_EF_CSL(0, 500, 5)), (y_obs_NoEF, y_obs_EF))

end