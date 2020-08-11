module ElectroTaxisSingleCellAnalysis

using ..ElectroTaxis
using ..LikelihoodFree
using Distributions, LinearAlgebra, IndexedTables, Combinatorics, InvertedIndices
import .LikelihoodFree.domain
import .LikelihoodFree.ndims


const prior_support = [(0.001, 3.0) (0.001, 5.0) (0.001, 5.0) (0.001, 1.0) (0.0, 2.0) (0.0, 2.0) (0.0, 2.0) (0.0,2.0)]
const prior_flat_components = Dict(
    :base => DistributionGenerator(SingleCellModel, product_distribution(vec([Uniform(interval...) for interval in prior_support[1:4]]))),
    :g1 => DistributionGenerator(VelocityBias, Uniform(prior_support[5]...)),
    :g2 => DistributionGenerator(SpeedIncrease, Uniform(prior_support[6]...)),
    :g3 => DistributionGenerator(SpeedAlignment, Uniform(prior_support[7]...)),
    :g4 => DistributionGenerator(PolarityBias, Uniform(prior_support[8]...)),
)

# ALL POSSIBLE MODELS FROM NoEF TO ALL BIASES
all_labels = [[:base, lab...] for lab in powerset([:g1, :g2, :g3, :g4])]
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

export prior_NoEF, prior_Full, prior_Best
const prior_NoEF = prior_flat_all[:base]
const prior_Full = prior_flat_all[Symbol(:base, :g1, :g2, :g3, :g4)]
const prior_Best = prior_flat_all[Symbol(:base, :g1, :g2, :g4)]

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

const Σ_Joint_BSL_IS = ISProposal(prior_Best, (L_NoEF_BSL(500), L_EF_BSL(500)), (y_obs_NoEF, y_obs_EF))
const Σ_Joint_BSL_SMC = SMCWrapper(
    prior_Best,
    Tuple(((L_NoEF_BSL(n), L_EF_BSL(n)), (y_obs_NoEF, y_obs_EF)) for n in 50:50:500)
)

######## Joint (Sequential)
# The following are functions because they depend on previously simulated data

using Random, StatsBase
Random.seed!(1)
const test_idx_NoEF = sample(1:50, 10, replace=false)
const test_idx_EF = sample(1:50, 10, replace=false)
Random.seed!()

# Check every worker has the same set of idx
println(test_idx_NoEF)


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
    :base => posterior_NoEF(),
    :g1 => DistributionGenerator(VelocityBias, Uniform(prior_support[5]...)),
    :g2 => DistributionGenerator(SpeedChange, Uniform(prior_support[6]...)),
    :g3 => DistributionGenerator(SpeedAlignment, Uniform(prior_support[7]...)),
    :g4 => DistributionGenerator(PolarityBias, Uniform(prior_support[8]...)),
)

export prior_sequential_all, prior_sequential_full
function prior_sequential_all() 
    prior_components = prior_sequential_components()
    return Dict(form_generator(prior_components, k...) for k in all_labels)
end
prior_sequential_full() = prior_sequential_all()[Symbol(:base, :g1, :g2, :g3, :g4)]

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

    F = maximum(logw)
    w = exp.(logw .- F)
    return log(mean(w, wt)) + F
end

using JLD
function test_loglikelihood(fn::String="./applications/electro/test_loglikelihood.jld")
    T = load_Combinatorial()
    L = Dict(k=>test_loglikelihood(T[k]) for k in keys(T))
    save(fn, "L", L)
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

X_labels = Dict(
    :base => "∅",
    :baseg1 => "1",
    :baseg2 => "2",
    :baseg3 => "3",
    :baseg4 => "4",
    :baseg1g2 => "1, 2",
    :baseg1g3 => "1, 3",
    :baseg1g4 => "1, 4",
    :baseg2g3 => "2, 3", 
    :baseg2g4 => "2, 4",
    :baseg3g4 => "3, 4",
    :baseg1g2g3 => "1, 2, 3",
    :baseg1g2g4 => "1, 2, 4",
    :baseg1g3g4 => "1, 3, 4",
    :baseg2g3g4 => "2, 3, 4",
    :baseg1g2g3g4 => "1, 2, 3, 4",
)

using StatsPlots
export see_selection_xIC, see_selection_Bayes

function see_selection_xIC(; kwargs...)
    T = load_Combinatorial()
    lbl = [Symbol(sym...) for sym in all_labels]
    nam = [X_labels[k] for k in lbl]
#    ctg = repeat(["AIC", "BIC"], inner=length(lbl))

    AICVec = [sequential_AIC(T, k) for k in lbl]
    AIC_sort = sortperm(AICVec, rev=true)
    figA = bar(AICVec[AIC_sort]; xlim=(600,900), yticks=(1:16, nam[AIC_sort]), legend=:none, 
    orientation=:h,
    xlabel="AIC",
    ylabel="Parameter space X")

    BICVec = [sequential_BIC(T, k) for k in lbl]
    BIC_sort = sortperm(BICVec, rev=true)
    figB = bar(BICVec[BIC_sort]; xrotation=45, xlim=(600,900), yticks=(1:16, nam[BIC_sort]), legend=:none, 
    orientation=:h,
    xlabel="BIC")

    fig = plot(figA, figB; layout = (1,2), kwargs...)
end

function see_selection_Bayes(fn::String="./applications/electro/test_loglikelihood.jld"; kwargs...)
    test_loglikelihood = load(fn, "L")
    lbl = [Symbol(sym...) for sym in all_labels]
#    ctg = repeat(["AIC", "BIC"], inner=length(lbl))

    nam = [X_labels[k] for k in lbl]    
    L = [test_loglikelihood[k] for k in lbl]
    L .-= maximum(L)
    broadcast!(exp, L, L)
    L ./= sum(L)

    L_sort = sortperm(L)
    figA = bar(L[L_sort]; yticks=(1:16, nam[L_sort]), title="λ=0 (Uniform model prior)", legend=:none, 
    orientation=:h,
    xlabel="Posterior mass p_X",
    ylabel="Parameter space X")

    p2 = [test_loglikelihood[k] - 2*ndims(model_all[k]) for k in lbl]
    p2 .-= maximum(p2)
    broadcast!(exp, p2, p2)
    p2 ./= sum(p2)
#    broadcast!(log, p2, p2)

    p2_sort = sortperm(p2)
    figB = bar(p2[p2_sort]; yticks=(1:16, nam[p2_sort]), title="λ=2", legend=:none, 
    orientation=:h,
    xlabel="Posterior mass p_X")

    fig = plot(figA, figB; layout = (1,2), kwargs...)
end


################################ Compare to data
using StatsBase

export θbar, θgen, visualise
export compare_data_NoEF, compare_data_EF

function θbar(T)
    F(θ) = [values(θ)...]
    F_θ = F.(select(T, :θ))
    w = select(T, :weight)
    Θ = eltype(select(T, :θ))
    return Θ(sum(F_θ.*w)./sum(w))
end
function θgen(T)
    θ = select(T, :θ)
    w = Weights(select(T, :weight))
    return sample(θ, w)
end
θgen(θ::NamedTuple) = θ

function visualise(F, T, N) 
    fig = plot()
    for n in 1:N
        θ = θgen(T)
        sol_n = F(; θ..., output_trajectory=true)
        plot!(fig, broadcast(t->sol_n(t)[1], F.saveat))
    end
    plot!(fig; legend=:none, ratio=:equal, framestyle=:origin, xlabel="x", ylabel="y")
    return fig
end
function visualise(y_obs)
    fig = plot()
    for traj in y_obs
        plot!(fig, traj)
    end
    plot!(fig; legend=:none, ratio=:equal, framestyle=:origin, xlabel="x", ylabel="y")
    return fig
end

function compare_data_NoEF()
    Θ = merge(SingleCellModel)
    T = load_sample("./applications/electro/NoEF_BSL_SMC.jld", Θ)[end]
    θ = θbar(T)
    data_NoEF = visualise(NoEF_trajectories)
    common_NoEF = visualise(F_NoEF, θ, 50)
    dist_NoEF = visualise(F_NoEF, T, 50)
    plot(
        data_NoEF, 
        common_NoEF, 
        dist_NoEF, 
        layout=(1,3), 
        link=:all,
        size=(900,300),
    )
end


function compare_data_EF()
    Θ = merge(SingleCellModel, SpeedChange, PolarityBias, PositionBias)
    T = load_sample("./applications/electro/Joint_BSL_SMC.jld", Θ)[end]
    θ = θbar(T)
    data_NoEF = visualise(NoEF_trajectories)
    data_EF = visualise(EF_trajectories)
    common_NoEF = visualise(F_NoEF, θ, 50)
    common_EF = visualise(F_EF, θ, 50)
    dist_NoEF = visualise(F_NoEF, T, 50)
    dist_EF = visualise(F_EF, T, 50)

    fig_NoEF = plot(data_NoEF, common_NoEF, dist_NoEF, layout=(1,3), link=:all)
    fig_EF = plot(data_EF, common_EF, dist_EF, layout=(1,3), link=:all)
    fig = plot(
        fig_NoEF, 
        fig_EF, 
        layout=(2,1), 
        size=(900,600),
    )
end

################################ Analyse switch
using StatsPlots, Statistics
const EMF_switch = StepEF(complex(1), complex(-1), 90)
const EMF_stop = StepEF(complex(1), complex(0), 90)

export F_switch, F_stop, step_figs
const F_switch = SingleCellSimulator(σ_init=0.1, emf=EMF_switch)
const F_stop = SingleCellSimulator(σ_init=0.1, emf=EMF_stop)

function step_figs(F, T, N::Int64=5; kwargs...)
    
    fig_traj_1 = plot(legend=:none, link=:both, title="Displacement: 0 < t < 90min")
    fig_traj_2 = plot(legend=:none, link=:both, title="Displacement: 90 < t < 180min")
    fig_polarity_magnitude = plot(legend=:none, xticks=[0, 90, 180], title="Polarity: Magnitude", ylabel="abs(p)")
    fig_polarity_angle = plot(xticks=[0, 90, 180], title="Polarity: Aligned Component", ylabel="cos(arg(p))")
    fig_polarity_sinangle = plot(xticks=[0, 90, 180], title="Polarity: Perpendicular Component", xlabel="t", ylabel="abs(sin(arg(p)))")

    sol = [F(; θgen(T)..., output_trajectory=true) for n in 1:(100*N)]
    
    for n in 1:N
        fun_traj(t) = sol[n](t)[1]
        fun_polarity_magnitude(t) = abs(sol[n](t)[2])
        fun_polarity_angle(t) = fun_polarity_magnitude(t)<0.6 ? NaN : cos(angle(sol[n](t)[2])) 
        fun_polarity_sinangle(t) = fun_polarity_magnitude(t)<0.6 ? NaN : abs(sin(angle(sol[n](t)[2]))) 

        plot!(fig_traj_1, fun_traj.(0:0.1:90), ratio=:equal, alpha=0.5, label="")
        plot!(fig_traj_2, fun_traj.(90:0.1:180).-fun_traj(90), ratio=:equal, alpha=0.5, label="")
        plot!(fig_polarity_magnitude, 0:1:180, fun_polarity_magnitude, alpha=0.5, label="")
        plot!(fig_polarity_angle, 0:1:180, fun_polarity_angle, alpha=0.5, label="")
        plot!(fig_polarity_sinangle, 0:1:180, fun_polarity_sinangle, alpha=0.5, label="")
    end

    plot!(fig_polarity_magnitude, 0:1:180, t->abs(F.emf(t)), seriescolor=:black, label="EF Input")
    plot!(fig_polarity_angle, 0:1:180, t->iszero(F.emf(t)) ? 0.0 : cos(angle(F.emf(t))), seriescolor=:black, label="EF Input")
    plot!(fig_polarity_sinangle, 0:1:180, t->iszero(F.emf(t)) ? 0.0 : abs(sin(angle(F.emf(t)))), seriescolor=:black, label="EF Input")

    mean_traj(t) = mean([sol_n(t)[1] for sol_n in sol])
    mean_polarity_magnitude(t) = mean([abs(sol_n(t)[2]) for sol_n in sol])
    mean_polarity_angle(t) = mean([cos(angle(sol_n(t)[2])) for sol_n in sol if abs(sol_n(t)[2])>0.6])
    mean_polarity_sinangle(t) = mean([abs(sin(angle(sol_n(t)[2]))) for sol_n in sol if abs(sol_n(t)[2])>0.6])

    plot!(fig_traj_1, mean_traj.(0:0.1:90), ratio=:equal, seriescolor=:red, label="Ensemble average")
    plot!(fig_traj_2, mean_traj.(90:0.1:180).-mean_traj(90), ratio=:equal, seriescolor=:red, label="Ensemble average")
    plot!(fig_polarity_magnitude, 0:1:180, mean_polarity_magnitude, seriescolor=:red, label="Ensemble average")
    plot!(fig_polarity_angle, 0:1:180, mean_polarity_angle, seriescolor=:red, label="Ensemble average", legend=:none)
    plot!(fig_polarity_sinangle, 0:1:180, mean_polarity_sinangle, seriescolor=:red, label="Ensemble average", legend=:left)

#    endpoints_1 = [sol_n(90)[1] for sol_n in sol]
#    endpoints_2 = [sol_n(180)[1]-sol_n(90)[1] for sol_n in sol]

#    plot!(fig_traj_1, real.(endpoints_1), imag.(endpoints_1), 
#        seriestype=:scatter, markershape=:xcross, markeralpha=0.3, markersize=1, markercolor=:red, label="")
#    plot!(fig_traj_2, real.(endpoints_2), imag.(endpoints_2), 
#        seriestype=:scatter, markershape=:xcross, markeralpha=0.3, markersize=1, markercolor=:red, label="")

    fig_traj = plot(fig_traj_1, fig_traj_2, layout = (1,2), link=:all, ratio=:equal, xlabel="x", ylabel="y")
    return plot(fig_traj, fig_polarity_magnitude, fig_polarity_angle, fig_polarity_sinangle;
    layout=(4,1), size=(800, 600), kwargs...)
end

export coarse
using SparseArrays
function coarse(z::Complex)
    if abs(z)<0.6
        return 1
    else
        re = real(z)
        im = imag(z)
        if abs(re)<abs(im)
            return 3
        else
            return re>0 ? 2 : 4
        end
    end
end
function coarse(z1::Complex, z2::Complex)
    mat = zeros(Int64, 4, 4)
    coarse!(mat, z1, z2)
    return mat
end
function coarse(sol, t::Real)
    vec = zeros(Int64, 4)
    coarse!(vec, sol, t)
    return vec
end
function coarse(sol, t1::Real, t2::Real)
    mat = zeros(Int64, 4, 4)
    coarse!(mat, sol, t1, t2)
    return mat
end

function coarse!(mat, z1::Complex, z2::Complex)
    i = coarse(z1)
    j = coarse(z2)
    mat[i, j] += 1
    return nothing
end
function coarse!(vec, sol, t::Real)
    for sol_n in sol
        i = coarse(sol_n(t)[2])
        vec[i] += 1
    end
    return nothing
end
function coarse!(mat, sol, t1::Real, t2::Real)
    for sol_n in sol
        coarse!(mat, sol_n(t1)[2], sol_n(t2)[2])
    end
    return nothing
end

export coarse_step_figs
function coarse_step_figs(F, T, N::Int64=1000; dt::Float64=5.0, kwargs...)

    state_lbl = ["Depolarised", "Polarised positive", "Polarised perpendicular", "Polarised negative"]
    state_clr = [1, 3, 4, 2]

    sol = [F(; θgen(T)..., output_trajectory=true) for n in 1:N]
    fig_states = plot(; title="State")
    fig_to4 = plot(; title="Net flux to ($(state_lbl[4]))")
    fig_from2 = plot(; title="Net flux from ($(state_lbl[2]))")
    fig_from1 = plot(; title="Net flux from ($(state_lbl[1]))")

    function f(i::Int64, j::Int64, t::Float64) 
        mat = coarse(sol, t-dt, t)
        return mat[i, j]-mat[j,i]
    end
    function f(i::Int64, t::Float64)
        vec = coarse(sol, t)
        return vec[i]
    end

    tvec = 70:dt:180

    for i in 1:3
        plot!(fig_to4, tvec, t->f(i, 4, t), label="from $(state_lbl[i])", seriescolor=state_clr[i])
    end
    for j in [1,3,4]
        plot!(fig_from2, tvec, t->f(2, j, t), label="to $(state_lbl[j])", seriescolor=state_clr[j])
    end
    for j in 2:4
        plot!(fig_from1, tvec, t->f(1, j, t), label="to $(state_lbl[j])", seriescolor=state_clr[j])
    end
    for k in 1:4
        plot!(fig_states, 0.0:1.0:180, t->f(k, t), label="$(state_lbl[k])", seriescolor=state_clr[k])
    end

    fig = plot(
        fig_states, 
        coarse_state(),
        fig_from2, 
        #fig_from1, 
        fig_to4;
     layout=(2,2), xticks=0:30:180, size=(800,800), legendfontsize=6, titlefontsize=8, kwargs...)
    return fig
end

export coarse_state
function coarse_state()
    P2 = StatsPlots.Plots.P2
    X1 = StatsPlots.Plots.partialcircle(0, 2π, 100, 0.6);
    X2 = P2[(0,0), (5,5), (5,-5)]
    X3a = P2[(0,0), (5,5), (-5,5)]
    X3b = P2[(0,0), (-5,-5), (5,-5)]
    X4 = P2[(0,0), (-5, 5), (-5, -5)]
    
    fig = plot(; ratio=:equal, xlims=(-1.2, 1.2), ylims=(-1.2, 1.2), xlabel="x", ylabel="y", title="Coarse-grained Polarity")

    plot!(fig, Shape(X2), fillcolor=3, label="Polarised positive")
    plot!(fig, Shape(X3a), fillcolor=4, label="")
    plot!(fig, Shape(X3b), fillcolor=4, label="Polarised perpendicular")
    plot!(fig, Shape(X4), fillcolor=2, label="Polarised negative")
    plot!(fig, Shape(X1), fillcolor=1, label="Depolarised")
    plot!(fig, sin, cos, 0, 2π, color=:black, label="", linestyle=:dash)

    return fig
end

export see_stationary_velocity
function see_stationary_velocity(; kwargs...)
    fig = plot()
    
    heatmap!(fig, -4:0.05:4, -4:0.05:4, (x,y)->stationary_velocity(complex(x,y); kwargs...);
        ratio=:equal,
        c=:Blues_9,
        xlims = (-4,4),
        ylims = (-4,4),
        grid=:false,
        colorbar=:false,
        tick_direction=:out,
        framestyle=:box,
        kwargs...
        )
    return fig
end

export compare_stationary_velocity
function compare_stationary_velocity()

    pars = (polarised_speed = 1.0, EB_on = 1.5, EB_off=1.0, σ=0.25,
    position_bias = 0.5,
    speed_change = 1.0,
    alignment_bias = 1.0,
    polarity_bias = 0.5,
    )

    EMF_off = NoEF()
    EMF_on = ConstantEF(complex(1.0))

    fig_off = see_stationary_velocity(; pars..., EMF=EMF_off)
    fig_on = see_stationary_velocity(; pars..., EMF=EMF_on)

    v = pars.polarised_speed
    γ1 = pars.position_bias
    γ2 = pars.speed_change
    γ3 = pars.alignment_bias
    γ4 = pars.polarity_bias

    plot!(fig_off; title="Autonomous model", 
    xticks = ([0, 1].*v, ["0","v"]),
    yticks = ([0, 1].*v, ["0","v"]),
    )
    plot!(fig_on; title="Electrotactic model",
    xticks = (([-(1+γ2-γ3), 0, (1+γ2+γ3)].*v) .+ (γ1*v), ["γ1 v - (1+γ2-γ3)v", "γ1 v","γ1 v + (1+γ2+γ3)v"]),
    yticks = ([0, 1].*v*(1+γ2), ["0","(1+γ2) v"]),
    )

    line_opt = (label="", c=:black,)
    hline!(fig_on, [0.0]; line_opt...)
    vline!(fig_on, [0.0]; line_opt...)
    hline!(fig_off, [0.0]; line_opt...)
    vline!(fig_off, [0.0]; line_opt...)

    plot!(fig_on, xrotation=45)
    fig = plot(fig_off, fig_on, layout=(1,2), legend=:none, tickfontsize=6, tickfonthalign=:right)
    return fig

end

end