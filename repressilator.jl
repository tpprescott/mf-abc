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

x0 = [0.0; 0.0; 0.0; 40.0; 20.0; 60.0]
ko = (1.0, 2.0, 5.0, 1000.0, 20.0)
T = 10.0