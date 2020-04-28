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
const L_NoEF = BayesianSyntheticLikelihood(F_NoEF, num_simulations=500)
const Σ_NoEF = MonteCarloProposal(prior_NoEF, K_NoEF, L_NoEF, y_obs_NoEF)

test_NoEF = first(Σ_NoEF)

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
const L_EF = BayesianSyntheticLikelihood(F_EF, num_simulations=500)
const Σ_EF = MonteCarloProposal(prior_EF, K_EF, L_EF, y_obs_EF)

test_EF = first(Σ_EF)

export F1, F2

function F1()
    return collect(Iterators.take(Σ_NoEF, 1000))
end
function F2()
    return collect(Iterators.take(Σ_EF, 1000))
end

end