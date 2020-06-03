module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra, IndexedTables, Combinatorics

const prior_support = [(0.001, 3) (0,5) (0,5) (0.001, 2) (0,2) (0,2) (0,2) (0,2)]
const bias_model_components = [SpeedChange, PolarityBias, PositionBias, AlignmentBias]
const bias_prior_components = broadcast((x,y)->DistributionGenerator(x, Uniform(y...)), bias_model_components, prior_support[5:8])

export prior_NoEF, y_obs_NoEF, F_NoEF
const prior_NoEF = DistributionGenerator(SingleCellModel, product_distribution(vec([
    Uniform(interval...) for interval in prior_support[1:4]
])))
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)

export prior_EF_powerset, y_obs_EF, F_EF
const prior_EF_powerset = [ProductGenerator(prior_NoEF, prior_combination...) for prior_combination in powerset(bias_prior_components)]
const y_obs_EF = EF_displacements
const EMF = ConstantEF(1.0)
const F_EF = SingleCellSimulator(σ_init=0.1, emf = EMF)

export prior_Biases, prior_FullModel
const prior_Biases = ProductGenerator(bias_prior_components...)
const prior_FullModel = ProductGenerator(prior_NoEF, prior_Biases)

######### NoEF

export Σ_NoEF_BSL_IS, Σ_NoEF_BSL_SMC

L_NoEF_BSL(n) = BayesianSyntheticLikelihood(F_NoEF, numReplicates=n)
const Σ_NoEF_BSL_IS = ISProposal(prior_NoEF, L_NoEF_BSL(500), y_obs_NoEF)
const Σ_NoEF_BSL_SMC = SMCWrapper(
    prior_NoEF,
    Tuple(((L_NoEF_BSL(n),), (y_obs_NoEF,)) for n in 50:50:500)
    ) 

######## EF Alone

export Σ_EF_BSL_IS, Σ_EF_BSL_SMC

L_EF_BSL(n) = BayesianSyntheticLikelihood(F_EF, numReplicates=n)
const Σ_EF_BSL_IS = ISProposal(prior_FullModel, L_EF_BSL(500), y_obs_EF)
const Σ_EF_BSL_SMC = SMCWrapper(
    prior_FullModel,
    Tuple(((L_EF_BSL(n),), (y_obs_EF,)) for n in 50:50:500)
)

######## Joint (Simultaneous)

export Σ_Joint_BSL_IS, Σ_Joint_BSL_SMC

const Σ_Joint_BSL_IS = ISProposal(prior_FullModel, (L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_SMC = SMCWrapper(
    prior_FullModel,
    Tuple(((L_NoEF_BSL(n), L_EF_BSL(n)), (y_obs_NoEF, y_obs_EF)) for n in 50:50:500)
)

######## Joint (Sequential)
# The following are functions because they depend on previously simulated data

export prior_Sequential
function prior_Sequential(prior_bias::AbstractGenerator = prior_Biases)
    t = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel)
    q = SequentialImportanceDistribution(t[end], prior_NoEF) # Note this forces the support of q equal to that of prior_NoEF
    return ProductGenerator(q, prior_bias)
end
function prior_Sequential(prior_bias::AbstractGenerator...)
    t = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel)
    q = SequentialImportanceDistribution(t[end], prior_NoEF) # Note this forces the support of q equal to that of prior_NoEF
    return [ProductGenerator(q, π) for π in prior_bias]
end
prior_Sequential(x) = prior_Sequential(x...)

export Σ_Sequential_BSL_IS, Σ_Sequential_BSL_SMC
function Σ_Sequential_BSL_IS(prior)
    return ISProposal(prior, (L_EF_BSL(500),), (y_obs_EF,))
end
function Σ_Sequential_BSL_SMC(prior)
    return SMCWrapper(
        prior,
        Tuple(((L_EF_BSL(n),), (y_obs_EF,)) for n in 50:50:500)
    )
end

_join(a::Symbol, b::Symbol) = Symbol(a,:_,b)
_join(a::Symbol, b::Symbol, c::Symbol...) = _join(_join(a,b), c...)
_join(a::Symbol) = a
_join() = :_
# Deal with all possible mechanisms
function infer_all_models()
    for prior in prior_EF_powerset
        fn_bits = fieldnames(domain(prior))[5:end]
        fn_id = _join(fn_bits...)
        fn = "./applications/electro/EF_Combinatorial_"*String(fn_id)*"jld"
        Σ = Σ_Sequential_BSL_SMC(prior)
        t = smc_sample(Σ, Tuple(1000:1000:10000), Tuple(2.0 .^ (-9:1:0)))
        save_sample(fn, t)
    end
    return nothing
end

########## I/O Functions
export see_parameters_NoEF, see_parameters_Joint
function see_parameters_NoEF(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel)
    T = t[generation]
    C = (cols===nothing) ? (1:4) : cols
    fig = parameterscatter(filter(r->r.weight>0, T), xlim=prior_support[C]; columns=C, kwargs...)
end

function see_parameters_Joint(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/Joint_BSL_SMC.jld", merge(SingleCellModel, SingleCellBiases))
    T = t[generation]
    C = cols===nothing ? (1:8) : cols
    fig = parameterscatter(filter(r->r.weight>0, T), xlim=prior_support[C]; columns=C, kwargs...)
end

function see_parameters_Sequential(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/Sequential_BSL_SMC.jld", merge(SingleCellModel, SingleCellBiases))
    T = t[generation]
    C = cols===nothing ? (1:8) : cols
    fig = parameterscatter(filter(r->r.weight>0, T), xlim=prior_support[C]; columns=C, kwargs...)
end

end