import .LikelihoodFree.simulate
export simulate, _simulate

function _simulate(F::MFABCSimulatorNode{M, U, Y, W}, m::M, u::U, args::NTuple{N,NoiseInput})::Tuple{Y,Tuple{W,Vararg{NoiseInput,N}}} where M where U where Y where W where N
    w = W()
    y = _simulate(F, m, u, w, args)
    return y, (w, args...)
end
_simulate(F, m, u) = _simulate(F, m, u, ())

function _simulate(F::MFABCSimulatorNode{M, U, Y, W}, m::M, u::U, w::W, args::NTuple{N, NoiseInput})::Y where M where U where Y where W where N
    _couple!(w, args[end])
    return _simulate(F, m, u, w, args[1:end-1])
end
function _simulate(F::MFABCSimulatorNode{M, U, Y, W}, m::M, u::U, w::W, args::NTuple{0, NoiseInput})::Y where M where U where Y where W
    return _simulate(F, m, u, w)
end
function _simulate(F::MFABCSimulatorNode{M, U, Y, W}, m::M, u::U, w::W)::Y where M where U where Y where W
    return F(m,u,w)
end

function _couple!(d::Dst, s::Src) where Dst<:NoiseInput where Src<:NoiseInput
    # Default is to have no effect at all on w: should import and specify function for specific types to allows for c to mutate w
end

function simulate(MFF::MFABCSimulator{M,U,Y,Nv}, m::M, u::U)::Y where M where U where Y where Nv
    output = Y()
    cont = true
    v = findfirst(rand().<MFF.CDF_init)
    eta = one(Float64)
    c = ()
    while cont
        push!(output.eta, eta)
        push!(output.seq, v)
        yv, c = _simulate(MFF.F[v], m, u, c)
        push!(output.y, yv)
        cont, v, eta = next_simulation(MFF.H, m, u, output)
    end
    return output
end

include("next_simulation.jl")
