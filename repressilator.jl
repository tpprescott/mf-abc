module Repressilator

using MultiFidelityABC
using LinearAlgebra
using Dierckx
using Random

# Repressilator model specification contains:
# - Model in repressilator_model_generator - what to simulate
# - mf-abc specification (what is low-fidelity, high-fidelity, etc) in repressilator_mfabc_problem
# - Prior parameter distribution in repressilator_prior

const z =  zeros(3,3)
const nu = vcat(hcat(I,-I,z,z),hcat(z,z,I,-I))
const x0 = [0.0; 0.0; 0.0; 40.0; 20.0; 60.0]
const k_nominal = (1.0, 2.0, 5.0, 1000.0, 20.0)
const T = 10.0

function propensity!(v::Array{Float64,1}, x::Array{Float64,1}, k::NTuple{5,Float64})
                
    v[1:3] = k[1] .+ (k[4]*(k[5]^k[2]))./((k[5]^k[2]) .+ (x[[6,4,5]].^k[2]))
    v[4:6] = x[1:3]
    v[7:9] = k[3].*x[1:3]
    v[10:12] = k[3].*x[4:6]
        
    return nothing
end

function diff_propensity!(dv::Array{Float64,2}, x::Array{Float64,1}, k::NTuple{5,Float64})

    dv[1,6] = -k[4]*k[2].*(k[5].^k[2]).*(x[6].^(k[2]-1))*(k[5].^k[2] + x[6].^k[2]).^(-2);
    dv[2,4] = -k[4]*k[2].*(k[5].^k[2]).*(x[4].^(k[2]-1))*(k[5].^k[2] + x[4].^k[2]).^(-2);
    dv[3,5] = -k[4]*k[2].*(k[5].^k[2]).*(x[5].^(k[2]-1))*(k[5].^k[2] + x[5].^k[2]).^(-2);
    dv[4:6,1:3] = Matrix{Float64}(I,3,3);
    dv[7:9,1:3] = k[3]*Matrix{Float64}(I,3,3);
    dv[10:12,4:6] = k[3]*Matrix{Float64}(I,3,3);

    return nothing 
end

gm = GillespieModel(nu, propensity!, x0, k_nominal, T)
tlm = TauLeapModel(nu, propensity!, x0, k_nominal, T, diff_propensity!, 0.01, 3.0, 0.01)

function summary_statistics(t::Times, x::States)
    return vcat([Spline1D(t,x[j,:];k=1)(0:T) for j in 1:size(x,1)]...)
end

Random.seed!(123)
t, x = simulate(gm) # Simulate the nominal model with a fixed seed
Random.seed!()
yo = summary_statistics(t, x)

function distance(y::Array{Float64,1})
    return norm(y-yo)/T
end

function lofi(k::Parameters)
    t, x, c_pp = simulate(tlm, k)
    return distance(summary_statistics(t,x)), c_pp
end

# The following hifi *couples* low fidelity and high fidelity simulations:
function hifi(k::Parameters, c_pp::Coarse_PP)
    t, x = complete(tlm, k, c_pp)
    return distance(summary_statistics(t,x))
end

# # This hifi version would produce independent (uncoupled) high fidelity simulations:
# function hifi(k,pass)
#     gm = GillespieModel(rep_mg, k)
#     t, x = gillespieDM(gm)
#     return summary_statistics(t,x)
# end
    
using Distributions
function prior()
    k2 = rand(Uniform(1,4))
    k5 = rand(Uniform(10,30))
    return k_nominal[1], k2, k_nominal[3], k_nominal[4], k5
end

end