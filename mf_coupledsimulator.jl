export NoiseInput, AbstractCoupledSimulator
abstract type NoiseInput end
abstract type AbstractCoupledSimulator{M, U, Y, N<:NoiseInput} <: AbstractSimulator{M, U, Y} end
noise_type(::AbstractCoupledSimulator{M,U,Y,N}) where M where U where Y where N = N

import .LikelihoodFree.simulate

function simulate(F::AbstractCoupledSimulator{M,U,Y,N}, m::M, u::U, n::N)::Y where M where U where Y where N
    return F(m,u,n)
end
function simulate!(yy::AbstractArray{Y}, F::AbstractCoupledSimulator{M,U,Y,N}, mm::AbstractArray{M}, u::U, nn::AbstractArray{N})::Nothing where M where U where Y where N
    for i in eachindex[nn]
        yy[i] = simulate(F, mm[i], u, nn[i])
    end
    return nothing
end
function simulate(F::AbstractCoupledSimulator{M,U,Y,N}, mm::AbstractArray{M}, u::U, nn::AbstractArray{N})::Array{Y} where M where U where Y where N
    yy = Array{Y}(undef, size(mm))
    simulate!(yy, F, mm, u, nn)
    return yy
end