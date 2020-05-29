module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra, IndexedTables

const EMF = ConstantEF(1.0)

const prior_support = [(0.001, 3) (0,5) (0,5) (0.001, 2) (0,2) (0,2) (0,2) (0,2)]

# NoEF
const prior_NoEF = DistributionGenerator(SingleCellModel_NoEF, product_distribution(vec([
    Uniform(int...) for int in prior_support[1:4]
])))
const K_NoEF = PerturbationKernel{SingleCellModel_NoEF}(
    MvNormal(zeros(4), diagm(0=>fill(0.01, 4)))
)
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)
L_NoEF_BSL(n) = BayesianSyntheticLikelihood(F_NoEF, numReplicates=n)
L_NoEF_CSL(n, i) = BayesianSyntheticLikelihood(F_NoEF, numReplicates=n, numIndependent=i)


export Σ_NoEF_BSL_MC, Σ_NoEF_BSL_IS, Σ_NoEF_BSL_SMC, Σ_NoEF_CSL_MC
const Σ_NoEF_BSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_BSL(500), y_obs_NoEF)
const Σ_NoEF_BSL_IS = ISProposal(prior_NoEF, L_NoEF_BSL(500), y_obs_NoEF)
const Σ_NoEF_BSL_SMC = SMCWrapper(
    prior_NoEF,
    (
        ((L_NoEF_BSL( 50),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(100),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(150),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(200),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(250),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(300),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(350),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(400),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(450),), (y_obs_NoEF,)),
        ((L_NoEF_BSL(500),), (y_obs_NoEF,)),
    )
)
const Σ_NoEF_CSL_MC = MCMCProposal(prior_NoEF, K_NoEF, L_NoEF_CSL(500, 5), y_obs_NoEF)


# EF
const EMF = ConstantEF(1.0)
const prior_EF = DistributionGenerator(SingleCellModel_EF, product_distribution(vec([
    Uniform(int...) for int in prior_support
])))
const K_EF = PerturbationKernel{SingleCellModel_EF}(
    MvNormal(zeros(8), diagm(0=>fill(0.01, 8)))
)
const y_obs_EF = EF_displacements
const F_EF = SingleCellSimulator(σ_init=0.1, emf = EMF)
L_EF_BSL(n) = BayesianSyntheticLikelihood(F_EF, numReplicates=n)
L_EF_CSL(n,i) = BayesianSyntheticLikelihood(F_EF, numReplicates=n, numIndependent=i)

export Σ_EF_BSL_MC, Σ_EF_BSL_IS, Σ_EF_BSL_SMC, Σ_EF_CSL_MC
const Σ_EF_BSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_BSL(500), y_obs_EF)
const Σ_EF_BSL_IS = ISProposal(prior_EF, L_EF_BSL(500), y_obs_EF)
const Σ_EF_BSL_SMC = SMCWrapper(
    prior_EF,
    (
        ((L_EF_BSL( 50),), (y_obs_EF,)),
        ((L_EF_BSL(100),), (y_obs_EF,)),
        ((L_EF_BSL(150),), (y_obs_EF,)),
        ((L_EF_BSL(200),), (y_obs_EF,)),
        ((L_EF_BSL(250),), (y_obs_EF,)),
        ((L_EF_BSL(300),), (y_obs_EF,)),
        ((L_EF_BSL(350),), (y_obs_EF,)),
        ((L_EF_BSL(400),), (y_obs_EF,)),
        ((L_EF_BSL(450),), (y_obs_EF,)),
        ((L_EF_BSL(500),), (y_obs_EF,)),
    )
)
const Σ_EF_CSL_MC = MCMCProposal(prior_EF, K_EF, L_EF_CSL(500, 5), y_obs_EF)

# Joint
export Σ_Joint_BSL_MC, Σ_Joint_BSL_IS, Σ_Joint_CSL_MC, Σ_Joint_BSL_SMC
const Σ_Joint_BSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_IS = ISProposal(prior_EF, (L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_SMC = SMCWrapper(
    prior_EF,
    (
        ((L_NoEF_BSL( 50), L_EF_BSL( 50)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(100), L_EF_BSL(100)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(150), L_EF_BSL(150)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(200), L_EF_BSL(200)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(250), L_EF_BSL(250)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(300), L_EF_BSL(300)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(350), L_EF_BSL(350)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(400), L_EF_BSL(400)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(450), L_EF_BSL(450)), (y_obs_NoEF, y_obs_EF)),
        ((L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF)),
    )
)
const Σ_Joint_CSL_MC = MCMCProposal(prior_EF, K_EF, (L_NoEF_CSL(500 , 5), L_EF_CSL(500, 5)), (y_obs_NoEF, y_obs_EF))

export see_parameters_NoEF, see_parameters_Joint
function see_parameters_NoEF(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel_NoEF)
    T = t[generation]
    C = (cols===nothing) ? (1:4) : cols
    fig = parameterscatter(filter(r->r.weight>0, T), xlim=prior_support[C]; columns=C, kwargs...)
end

function see_parameters_Joint(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/Joint_BSL_SMC.jld", SingleCellModel_EF)
    T = t[generation]
    C = cols===nothing ? (1:8) : cols
    fig = parameterscatter(filter(r->r.weight>0, T), xlim=prior_support[C]; columns=C, kwargs...)
end



end