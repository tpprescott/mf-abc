module Viral

using ..MultiFidelityABC
using Random
using StatsBase
using LinearAlgebra

# Viral model specification contains:
# - Model in viral_model_generator - what to simulate
# - mf-abc specification (what is low-fidelity, high-fidelity, etc) in viral_mfabc_problem
# - Prior parameter distribution in viral_prior

const nu = Float64.([0 1 0 -1 0 0 ; 1 -1 0 0 0 -1 ; 0 0 1 0 -1 -1; 0 0 0 0 0 1])

function propensity!(v::Array{Float64,1}, x::Array{Float64,1}, k::NTuple{6,Float64})
    v[:] = k.*[x[1], x[2], x[1], x[1], x[3], x[2]*x[3]]
    return nothing
end

const x0 = [1.0, 0.0, 0.0, 0.0]
const k_nominal = (1.0, 0.025, 100.0, 0.25, 1.9985, 7.5e-5)
const T = 200.0

gm = GillespieModel(nu, propensity!, x0, k_nominal, T)

############ Add in requirements for a hybrid model

const stochastic_reactions = [true,true,false,true,false,true]
function reduced_propensity!(v::Array{Float64,1}, x::Array{Float64,1}, k::NTuple{6,Float64})
    propensity!(v, x, k)
    v[:] .*= stochastic_reactions 
    return nothing
end

function deterministic_step(t,x,k)
    err = (k[3]/k[5])*x[1] - x[3]
    if abs(1/err)<1
        tau = (-1/k[5]) * log(1 - abs(1/err))
    else
        tau = T-t
        err = 0
    end
    return tau, [0, 0, sign(err), 0]
end

hm = HybridModel(nu, propensity!, x0, k_nominal, T, reduced_propensity!, deterministic_step)
pop_size = 10

############ Set up a multifidelity problem

Random.seed!(123)
t_pop, x_pop = simulate(gm, pop_size) # Simulate the nominal model with a fixed seed
Random.seed!()

infected_threshold = 3
derivative_horizon = 30.0

viral_count(x) = x[4,end]
infected(x) = (viral_count(x)>infected_threshold)
viral_output(x,t) = (viral_count(x) - x[4,searchsortedfirst(t, T-derivative_horizon)])/derivative_horizon
infection_time(x,t) = (infected(x) && t[searchsortedfirst(x[4,:], infected_threshold)])

function summary_statistics(t_pop::Array{Times,1}, x_pop::Array{States,1})
    inf_flag = infected.(x_pop)
    if mean(inf_flag)>0
        return mean(inf_flag), mean(viral_count.(x_pop)[inf_flag]), mean(viral_output.(x_pop, t_pop)[inf_flag]), mean(infection_time.(x_pop,t_pop)[inf_flag])
    else
        return 0.0, Inf64, Inf64, Inf64
    end
end
yo = summary_statistics(t_pop, x_pop)

function distance(pc_infected::Float64, viral_count::Float64, viral_output::Float64, infection_time::Float64)
    return sqrt((pc_infected - yo[1])^2 + ((viral_count/yo[2])-1)^2 + ((viral_output/yo[3])-1)^2 + ((infection_time/yo[4])-1)^2)
end

function lofi(k::Parameters)
    t_pop, x_pop, pp_pop = simulate(hm, k, pop_size)
    return distance(summary_statistics(t_pop, x_pop)...), pp_pop
end

# The following hifi *couples* low fidelity and high fidelity simulations:
function hifi(k::Parameters, coupling_input::Array{PP,1})
    t_pop, x_pop = complete(hm, k, coupling_input)
    return distance(summary_statistics(t_pop, x_pop)...)
end

# # This hifi version would produce independent (uncoupled) high fidelity simulations:
# function hifi(k::Parameters, coupling_input::Array{PP,1})
#     t_pop, x_pop = simulate(gm, k, pop_size)
#     return distance(summary_statistics(t_pop,x_pop))
# end

using Distributions
function prior()::Parameters
    return map(ki->ki*(1.5^rand(Uniform(-1,1))), k_nominal) # Scale nominal parameters by factor between 2/3 and 3/2
end

mf_prob = MFABC(prior, lofi, hifi)

end