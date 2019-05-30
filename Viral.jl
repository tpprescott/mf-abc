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

viral_count(x) = x[4,end]
infected(x) = (viral_count(x)>infected_threshold)
infection_time(x,t) = (infected(x) && t[searchsortedfirst(x[4,:], infected_threshold)]/T)

function summary_statistics(t_pop::Array{Times,1}, x_pop::Array{States,1})
    inf_flag = infected.(x_pop)
    if mean(inf_flag)>0
        return mean(inf_flag), log(2,mean(viral_count.(x_pop)[inf_flag])), mean(infection_time.(x_pop,t_pop)[inf_flag])
    else
        return 0.0, Inf64, Inf64
    end
end
yo = summary_statistics(t_pop, x_pop)

distance(y1, y2) = norm(y1.-y2)
distance(y) = distance(y, yo)

function lofi(k::Parameters)
    t_pop, x_pop, pp_pop = simulate(hm, k, pop_size)
    return distance(summary_statistics(t_pop, x_pop)), pp_pop
end

# The following hifi *couples* low fidelity and high fidelity simulations:
function hifi(k::Parameters, coupling_input::Array{PP,1})
    t_pop, x_pop = complete(hm, k, coupling_input)
    return distance(summary_statistics(t_pop, x_pop))
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