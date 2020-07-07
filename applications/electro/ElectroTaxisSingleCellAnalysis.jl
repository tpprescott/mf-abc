module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra, IndexedTables, Combinatorics, InvertedIndices
import .LikelihoodFree.domain
import .LikelihoodFree.ndims

const prior_support = [(0.001, 3.0) (0.0, 5.0) (0.0, 5.0) (0.001, 2.0) (0.0, 2.0) (0.0, 2.0) (0.0, 2.0) (0.0,2.0)]
const prior_flat_components = Dict(
    :SCM => DistributionGenerator(SingleCellModel, product_distribution(vec([Uniform(interval...) for interval in prior_support[1:4]]))),
    :Spe => DistributionGenerator(SpeedChange, Uniform(prior_support[5]...)),
    :Pol => DistributionGenerator(PolarityBias, Uniform(prior_support[6]...)),
    :Pos => DistributionGenerator(PositionBias, Uniform(prior_support[7]...)),
    :Ali => DistributionGenerator(AlignmentBias, Uniform(prior_support[8]...)),
)

# ALL POSSIBLE MODELS FROM NoEF TO ALL BIASES
all_labels = [[:SCM, lab...] for lab in powerset([:Spe, :Pol, :Pos, :Ali])]
form_generator(components::Dict, k1) = (Symbol(k1) => components[k1])
form_generator(components::Dict, k1, keys...) = (Symbol(k1,keys...) => ProductGenerator(components[k1], (components[k] for k in keys)...))

export model_all, prior_flat_all
const prior_flat_all = Dict(form_generator(prior_flat_components, k...) for k in all_labels)
const model_all = Dict(k => LikelihoodFree.domain(prior_flat_all[k]) for k in keys(prior_flat_all))

export y_obs_NoEF, F_NoEF, y_obs_EF, F_EF
const y_obs_NoEF = NoEF_displacements
const F_NoEF = SingleCellSimulator(σ_init=0.1)
const y_obs_EF = EF_displacements
const EMF = ConstantEF(1.0)
const F_EF = SingleCellSimulator(σ_init=0.1, emf = EMF)

export prior_NoEF, prior_Full
const prior_NoEF = prior_flat_all[:SCM]
const prior_Full = prior_flat_all[Symbol(:SCM, :Spe, :Pol, :Pos, :Ali)]

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
const Σ_EF_BSL_IS = ISProposal(prior_Full, L_EF_BSL(500), y_obs_EF)
const Σ_EF_BSL_SMC = SMCWrapper(
    prior_Full,
    Tuple(((L_EF_BSL(n),), (y_obs_EF,)) for n in 50:50:500)
)

######## Joint (Simultaneous)

export Σ_Joint_BSL_IS, Σ_Joint_BSL_SMC

const Σ_Joint_BSL_IS = ISProposal(prior_Full, (L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_SMC = SMCWrapper(
    prior_Full,
    Tuple(((L_NoEF_BSL(n), L_EF_BSL(n)), (y_obs_NoEF, y_obs_EF)) for n in 50:50:500)
)

######## Joint (Sequential)
# The following are functions because they depend on previously simulated data

const test_idx_NoEF = 1:10
const test_idx_EF = 1:10


function construct_posterior_NoEF(; test_idx = test_idx_NoEF)
    t1 = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel)
    q1 = SequentialImportanceDistribution(t1[end], prior_NoEF) # Note this forces the support of the posterior equal to that of prior_NoEF
    Σ = ISProposal(prior_NoEF, q1, L_NoEF_BSL(500), getindex(y_obs_NoEF, InvertedIndices.Not(test_idx)))
    t = importance_sample(Σ, 10000)
    save_sample("./applications/electro/Sequential_NoEF.jld", [t])
    return nothing
end

export posterior_NoEF
function posterior_NoEF()
    t = load_sample("./applications/electro/Sequential_NoEF.jld", SingleCellModel)
    q = SequentialImportanceDistribution(t[end], prior_NoEF)
    return q
end
prior_sequential_components() = Dict(
    :SCM => posterior_NoEF(),
    :Spe => DistributionGenerator(SpeedChange, Uniform(prior_support[5]...)),
    :Pol => DistributionGenerator(PolarityBias, Uniform(prior_support[6]...)),
    :Pos => DistributionGenerator(PositionBias, Uniform(prior_support[7]...)),
    :Ali => DistributionGenerator(AlignmentBias, Uniform(prior_support[8]...)),
)

export prior_sequential_all, prior_sequential_full
function prior_sequential_all() 
    prior_components = prior_sequential_components()
    return Dict(form_generator(prior_components, k...) for k in all_labels)
end
prior_sequential_full() = prior_sequential_all()[Symbol(:SCM, :Spe, :Pol, :Pos, :Ali)]

export Σ_Sequential_BSL_IS, Σ_Sequential_BSL_SMC
function Σ_Sequential_BSL_IS(prior = prior_sequential_full())
    return ISProposal(prior, (L_EF_BSL(500),), (y_obs_EF,))
end
function Σ_Sequential_BSL_SMC(prior = prior_sequential_full())
    return SMCWrapper(
        prior,
        Tuple(((L_EF_BSL(n),), (y_obs_EF,)) for n in 50:50:500)
    )
end

# Deal with all possible mechanisms
export infer_all_models
function infer_all_models(; test_idx=test_idx_EF)
    N = Tuple(1000:1000:10000)
    σ = Tuple(2.0 .^ (-9:1:0))

    for (id, prior) in prior_sequential_all()
        Σ = Σ_Sequential_BSL_SMC(prior)
        t = smc_sample(Σ, N, scale = σ, test_idx = (test_idx,))
        fn = "./applications/electro/EF_Combinatorial_"*String(id)*".jld"
        save_sample(fn, t)
    end
    return nothing
end

export load_Combinatorial
function load_Combinatorial()
    return Dict(id => load_Combinatorial(id, Θ) for (id, Θ) in model_all)
end
function load_Combinatorial(id::Symbol, ::Type{Θ}) where Θ
    fn = "./applications/electro/EF_Combinatorial_"*String(id)*".jld"
    t = load_sample(fn, Θ)
    return t[end]
end

export sequential_AIC, sequential_BIC
# The following functions are specific to the sequential inference task, since logp is the previous experiment's posterior (using a flat prior)
function sequential_AIC(T::Dict{Symbol, IndexedTable}, k::Symbol)
    d, logLhat = sequential_xIC(T, k)
    return AIC(d, logLhat)
end
function sequential_BIC(T::Dict{Symbol, IndexedTable}, k::Symbol)
    d, logLhat = sequential_xIC(T, k)
    return BIC(d, 100, logLhat)
end
function sequential_xIC(T::Dict{Symbol, IndexedTable}, k::Symbol)
    Θ = model_all[k]
    d = ndims(Θ)

    t = T[k]
    flat_prior = prior_flat_all[k]

    approx_NoEF_loglikelihood = select(t, :logp) - logpdf.(Ref(flat_prior), select(t, :θ))
    logLhat = maximum(sum.(select(t, :logww)) .+ approx_NoEF_loglikelihood)
    
    return d, logLhat
end
AIC(d, logLhat) = 2*d - 2*logLhat
BIC(d, n, logLhat) = d*log(n) - 2*logLhat

const test_conditioner = ((L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF[test_idx_NoEF], y_obs_EF[test_idx_EF]))

using Distributed, StatsBase, ProgressMeter
export posweight, test_loglikelihood

posweight = row -> row.weight>0
test_loglikelihood(row) = loglikelihood(test_conditioner, row.θ).logw
function test_loglikelihood(t::IndexedTable)
    tt = filter(posweight, t)
    logw = @showprogress pmap(test_loglikelihood, tt)
    wt = Weights(select(tt, :weight))
    return mean(logw, wt)
end

########## Writeup Functions
export see_parameters_NoEF, see_parameters_Joint, see_parameters_Sequential
function see_parameters_NoEF(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/NoEF_BSL_SMC.jld", SingleCellModel)
    T = t[generation]
    C = (cols===nothing) ? (1:4) : cols
    fig = parameterweights(filter(r->r.weight>0, T); xlim=prior_support[C], columns=C, kwargs...)
end

function see_parameters_Joint(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/Joint_BSL_SMC.jld", merge(SingleCellModel, SingleCellBiases))
    T = t[generation]
    C = cols===nothing ? (1:8) : cols
    fig = parameterweights(filter(r->r.weight>0, T); xlim=prior_support[C], columns=C, kwargs...)
end

function see_parameters_SequentialFull(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/Sequential_BSL_SMC.jld", merge(SingleCellModel, SingleCellBiases))
    T = t[generation]
    C = cols===nothing ? (1:8) : cols
    fig = parameterweights(filter(r->r.weight>0, T); xlim=prior_support[C], columns=C, kwargs...)
end

function see_parameters_SequentialBest(; generation::Int64=10, cols=nothing, kwargs...)
    t = load_sample("./applications/electro/EF_Combinatorial_SCMSpePolPos.jld", merge(SingleCellModel, SpeedChange, PolarityBias, PositionBias))
    T = t[generation]
    C = cols===nothing ? (1:7) : cols
    fig = parameterweights(filter(r->r.weight>0, T); xlim=prior_support[C], columns=C, kwargs...)
end

using StatsPlots
export see_model_selection
function see_model_selection(; kwargs...)
    T = load_Combinatorial()
    lbl = [Symbol(sym...) for sym in all_labels]
    nam = repeat(lbl, outer=2)
    ctg = repeat(["AIC", "BIC"], inner=length(lbl))
    AICVec = [sequential_AIC(T, k) for k in lbl]
    BICVec = [sequential_BIC(T, k) for k in lbl]
    fig = groupedbar(nam, hcat(AICVec, BICVec); group=ctg, xrotation=45, ylim=[800,1100], kwargs...)
end

end