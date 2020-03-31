export NoiseProcess, AbstractCoupledSimulator
abstract type NoiseProcess end
abstract type AbstractCoupledSimulator{M, U, Y, N <: NoiseProcess} <: AbstractSimulator{M, U, Y} end

noise_type(::AbstractCoupledSimulator{M,U,Y,N}) where M where U where Y where N = N

import .LikelihoodFree.simulate, .LikelihoodFree.simulate!

function initialise_noise_process(noise_process::Type{N}, path...) where N<:NoiseProcess
    # Default is uncoupled: the noise process that comes out is the default one for the simulator (i.e. type N)
    # Can extend this function for specific combinations of noise process that are coupled in
    return noise_process()
end

function realise_noise(F::AbstractCoupledSimulator{M,U,Y,N}, m::M, u::U, noise::N)::Y where M where U where Y where N
    return F(m,u,noise)
end

function simulate(F::AbstractCoupledSimulator{M,U,Y,N}, m::M, u::U, path...) where M where U where Y where N
    noise = initialise_noise_process(N, path...)
    y = realise_noise(F, m, u, noise)
    return (y = y, n = noise)
end
