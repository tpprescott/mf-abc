using LinearAlgebra

z =  repeat([0.],3,3)
nu = vcat(hcat(I,-I,z,z),hcat(z,z,I,-I))

function propensity(x::Array{Float64,1}, k::NTuple{5,Float64})
    
    tx(p::Float64) = k[1] + k[4]*(k[5]^k[2])/((k[5]^k[2]) + (p^k[2]))
    degm(m::Float64) = m
    tl(m::Float64) = k[3]*m
    degp(p::Float64) = k[3]*p
    
    tx_block = broadcast(tx, x[[6;4;5]])
    degm_block = broadcast(degm, x[1:3])
    tl_block = broadcast(tl, x[1:3])
    degp_block = broadcast(degp, x[4:6])
    
    return vcat(tx_block, degm_block, tl_block, degp_block)
end

function diff_propensity(x::Array{Float64,1}, k::NTuple{5,Float64})

    dp = zeros(12,6);

    dp[1,6] = -k[4]*k[2].*(k[5].^k[2]).*(x[6].^(k[2]-1))*(k[5].^k[2] + x[6].^k[2]).^(-2);
    dp[2,4] = -k[4]*k[2].*(k[5].^k[2]).*(x[4].^(k[2]-1))*(k[5].^k[2] + x[4].^k[2]).^(-2);
    dp[3,5] = -k[4]*k[2].*(k[5].^k[2]).*(x[5].^(k[2]-1))*(k[5].^k[2] + x[5].^k[2]).^(-2);
    dp[4:6,1:3] = Matrix{Float64}(I,3,3);
    dp[7:9,1:3] = k[3]*Matrix{Float64}(I,3,3);
    dp[10:12,4:6] = k[3]*Matrix{Float64}(I,3,3);

    return dp 
end

x0 = [0.0; 0.0; 0.0; 40.0; 20.0; 60.0]
ko = (1.0, 2.0, 5.0, 1000.0, 20.0)
T = 10.0

function prior_sample()
    k1 = ko[1]
    k2 = rand(Uniform(1,4))
    k3 = ko[3]
    k4 = ko[4]
    k5 = rand(Uniform(10,30))
    return k1,k2,k3,k4,k5
end

function summary_statistics(t::Array{Float64,1}, x::Array{Array{Float64,1},1})
    t_obs = 0:1:10
    idx = [findlast(t .<= to) for to in t_obs]
    y = x[idx]
    return y
end

function ss_distance(y1::Array{Array{Float64,1},1},y2::Array{Array{Float64,1},1},T::Float64)
    return norm(y2-y1)/T
end