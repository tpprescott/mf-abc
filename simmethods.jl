export Simulatable, Parameters, GillespieModel, TauLeapModel, HybridModel
export Times, States, Coupling, Coarse_PP, PP
export simulate, complete

using Random
using Distributions
using Printf

abstract type Simulatable end
Parameters = Tuple{Vararg{Float64}}

######## Output types
Times = Array{Float64,1}
States = Array{Float64,2}

abstract type Coupling end

struct Coarse_PP <: Coupling
    distance_intervals::Array{Float64,2}
    interval_firings::Array{Int64,2}
end
struct PP <: Coupling
    firing_distances::Array{Array{Float64,1},1}
end

######## Direct simulation

struct GillespieModel <: Simulatable
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity!::Function           # Propensity function
    x0::Array{Float64,1}            # Initial conditions
    k_nominal::Parameters           # Parameters to simulate - type has to match input type of par_prior
    T::Float64                      # Simulation time
end

function gillespie_update!(t_traj::Array{Float64,1}, x::Array{Float64,1}, a::Array{Float64}, gm::GillespieModel)
    a0 = sum(a)
    if a0>0
        append!(t_traj, t_traj[end]+rand(Exponential(1/a0)))
        x[:] += gm.nu[:, rand(Categorical(a./a0))]
    else
        append!(t_traj, gm.T)
    end
    return nothing
end

function simulate(gm::GillespieModel, k::Parameters)::Tuple{Times, States}

    (nx,nr) = size(gm.nu)
    
    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(gm.x0)
    x::Array{Float64,1} = copy(gm.x0)
    speeds::Array{Float64,1} = zeros(nr)

    while t_traj[end] < gm.T
        gm.propensity!(speeds, x, k)
        gillespie_update!(t_traj, x, speeds, gm)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx, nt))

end

######## Tau Leap Simulation
# Produces trajectory and coarse-grained description of underlying Poisson process
# The tau leap size adapts according to epsilon (if positive) to limit the rate of change
# of the propensity function
# The parameter nc defines reactions as critical if nc firings would send molecule 
# counts negative: then the reactions are not allowed to proceed according to tau leap.

struct TauLeapModel <: Simulatable
    nu::Matrix{Float64}         # Stoichiometric matrix
    propensity!::Function       # Propensity function
    x0::Array{Float64,1}        # Initial conditions
    k_nominal::Parameters       # Parameters to simulate - type has to match input type of propensity
    T::Float64                  # Simulation time
    diff_propensity!::Function  # Differentiated propensity function
    tau::Real                   # Base non-adaptive tau-leap (if epsilon is zero)
    nc::Real                    # How many firings away from going negative makes a reaction critical
    epsilon::Real               # Error parameter for adaptive tau leaping
end

function simulate(tlm::TauLeapModel, k::Parameters)::Tuple{Times, States, Coupling}
    
    function decide_critical!(critical::Array{Bool,1}, x::Array{Float64,1}, speeds::Array{Float64,1}, L::Array{Float64,1}, critnu::Array{Bool,2}, posscrit::Array{Bool,1}, tlm::TauLeapModel)
        for j in 1:length(critical)
            if posscrit[j]
                if speeds[j]==0
                    critical[j] = false
                else
                    L[j] = minimum(ceil.(x[critnu[:,j]]./abs.(tlm.nu[critnu[:,j],j])))
                    critical[j] = (L[j] <= tlm.nc)
                end
            end
        end
        return nothing
    end

    function largest_timestep(speeds::Array{Float64,1}, diffspeeds::Array{Float64,2}, critical::Array{Bool,1}, tlm::TauLeapModel)
        # largest_timestep(speeds, diffspeeds, critical, epsilon, tlm)
        
        nr = size(tlm.nu,2)
        
        if .&(critical...)
            return Inf64
        else
            F = diffspeeds[.!critical, :] * tlm.nu[:, .!critical]
            mu = F*speeds[.!critical]
            sigma2 = (F.^2)*speeds[.!critical]
            v1 = tlm.epsilon*sum(speeds)./abs.(mu)
            v2 = ((tlm.epsilon*sum(speeds))^2)./sigma2
            return minimum([v1;v2])
        end
    end

    function do_tau_leap!(t_traj, x, d, f::Array{Int64,1}, tauprime, tauprimeprime, speeds, critical, tlm)
        # do_tau_leap!(t_traj, x, d, f, tauprime, tauprimeprime, speeds, critical)
 
        nr = length(speeds)

        tau_adaptive = min(tauprime,tauprimeprime)
        d[:] = tau_adaptive*speeds
        
        if tauprimeprime < tauprime
            fired_critical = rand(Categorical((speeds[critical])/sum(speeds[critical])))
        else
            fired_critical = -1
        end

        for j in 1:nr
            if !critical[j]
                f[j] = rand(Poisson(d[j]))
            elseif j == fired_critical
                f[j] = 1
            else
                f[j] = 0
            end
        end
        append!(t_traj, t_traj[end]+tau_adaptive) 
        x[:] += tlm.nu*f
    end

    function backstep!(t_traj, x_traj, x)
        pop!(t_traj)
        x[:] = x_traj[end-(length(x)-1):end]
    end

    (nx, nr) = size(tlm.nu)

    t_traj::Array{Float64,1} = [0]
    x_traj::Array{Float64,1} = copy(tlm.x0)
    x::Array{Float64,1} = copy(tlm.x0)

    d_traj::Array{Float64,1} = zeros(nr)
    f_traj::Array{Int64,1} = zeros(nr)
    d::Array{Float64,1} = zeros(nr)
    f::Array{Int64,1} = zeros(nr)
    
    L::Array{Float64,1} = zeros(nr)
    speeds::Array{Float64,1} = zeros(nr)
    diffspeeds::Array{Float64,2} = zeros(nr, nx)

    critnu::Array{Bool,2} = (tlm.nu .< 0)
    critical::Array{Bool,1} = [minimum(tlm.nu[:,j])<0 for j in 1:nr]
    posscrit::Array{Bool,1} = [minimum(tlm.nu[:,j])<0 for j in 1:nr]
    
    
    while t_traj[end] < tlm.T
        tlm.propensity!(speeds, x, k)
        decide_critical!(critical, x, speeds, L, critnu, posscrit, tlm)

        if tlm.epsilon>0
            tlm.diff_propensity!(diffspeeds, x, k)
            tauprime = largest_timestep(speeds, diffspeeds, critical, tlm)
        else
            tauprime = tlm.tau
        end

        if |(critical...)
            tauprimeprime = rand(Exponential(1/sum(speeds[critical])))
        else
            tauprimeprime = Inf64
        end

        do_tau_leap!(t_traj, x, d, f, tauprime, tauprimeprime, speeds, critical, tlm)
        while .|((x.<0)...)
            backstep!(t_traj, x_traj, x)
            tauprime /= 2
            do_tau_leap!(t_traj, x, d, f, tauprime, tauprimeprime, speeds, critical, tlm)
        end

        append!(x_traj,x)
        append!(d_traj,d)
        append!(f_traj,f)

    end
    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx, nt)), Coarse_PP(reshape(d_traj,(nr, nt)), reshape(f_traj,(nr, nt)))

end

######## Hybrid deterministic/continuous simulation
# Produces trajectory and Poisson process of the stochastic firings

struct HybridModel <: Simulatable
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity!::Function           # Propensity function
    x0::Array{Float64,1}            # Initial conditions
    k_nominal::Parameters           # Parameters to simulate - type has to match input type of par_prior
    T::Float64                      # Simulation time
    reduced_propensity!::Function   # Reduced propensity function (only return speeds of stochastic reactions)
    deterministic_step::Function    # The deterministic reaction wait time and result
end

function simulate(hm::HybridModel, k::Parameters)::Tuple{Times, States, Coupling}

    (n_x, n_r) = size(hm.nu)

    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(hm.x0)
    x::Array{Float64,1} = copy(hm.x0)
    t::Float64 = 0.0
    
    d::Array{Float64,1} = zeros(n_r)
    d_next_event::Array{Float64,1} = randexp(n_r)
    pp_set::Array{Array{Float64,1},1} = [[d_i] for d_i in d_next_event]
    
    speeds::Array{Float64,1} = zeros(n_r)
    while t<hm.T
        hm.reduced_propensity!(speeds, x, k) # Calculate speeds (will be zero along deterministic reactions)
        
        tau_det, nu_j_det = hm.deterministic_step(t, x, k) # time until next deterministic event
        tau_sto, j_sto = findmin((d_next_event-d)./speeds)
        
        if tau_det <= tau_sto
            t += tau_det
            x += nu_j_det
            d += tau_det*speeds
        else
            t += tau_sto
            x += hm.nu[:,j_sto]
            d += tau_sto*speeds
            d_next_event[j_sto] += randexp()
            push!(pp_set[j_sto], d_next_event[j_sto])
        end
        
        if t<=hm.T
            push!(t_traj, t)
            append!(x_traj, x)
        else
            push!(t_traj, hm.T)
            append!(x_traj, x_traj[(end-n_x+1):end])
            n_t = length(t_traj)
            return t_traj, reshape(x_traj,(n_x, n_t)), PP(pp_set)
        end
    end

    return t_traj, reshape(x_traj,(n_x, length(t_traj))), PP(pp_set)

end

######## Coupling methods, based on
# (A) making a coarse-grained Poisson process fine-grained
# (B) mapping a known Poisson process to an exact trajectory (filling in gaps where needed)

function bridge_PP(c::Coarse_PP)::PP
    nr = size(c.distance_intervals,1)
    firings = c.distance_intervals[:,2:end].*rand.(c.interval_firings[:,2:end])
    D = cumsum(c.distance_intervals,dims=2)
    pp_split = broadcast((a,b)->a.+b, D[:,1:end-1], firings)
    sort!.(pp_split)
    pp_fired = [vcat(pp_split[j,:]...) for j = 1:nr]
    return PP(pp_fired)
end

function gillespie_update!(t_traj::Array{Float64,1}, x::Array{Float64,1}, next_event_d::Array{Float64, 1}, d::Array{Float64,1}, pps::Array{Array{Float64,1},1}, a::Array{Float64,1}, mdl::Simulatable)
    tau, j = findmin((next_event_d - d)./a)
    d[:] += a*tau
    if isempty(pps[j])
        next_event_d[j] += randexp()
    else
        next_event_d[j] = popfirst!(pps[j])
    end
    if isfinite(tau)
        append!(t_traj, t_traj[end]+tau)
        x[:] += mdl.nu[:,j]
    else
        append!(t_traj, mdl.T)
    end
end

function map_pp_to_trajectory(mdl::Simulatable, k::Parameters, pp::PP)::Tuple{Times, States}

    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(mdl.x0)
    x = copy(mdl.x0)
    
    (nx,nr) = size(mdl.nu)
    d = zeros(nr)
    speed = zeros(nr)

    next_event_d = randexp(nr)
    for j in 1:nr
        if ~isempty(pp.firing_distances[j])
            next_event_d[j] = popfirst!(pp.firing_distances[j])
        end
    end

    while t_traj[end] < mdl.T
        mdl.propensity!(speed, x, k)
        gillespie_update!(t_traj, x, next_event_d, d, pp.firing_distances, speed, mdl)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx,nt))
end


######### Apply coupling methods

function complete(tlm::TauLeapModel, k::Parameters, c_pp::Coarse_PP)::Tuple{Times, States}
    t_traj, x_traj = map_pp_to_trajectory(tlm, k, bridge_PP(c_pp))
end
function complete(hm::HybridModel, k::Parameters, pp::PP)::Tuple{Times, States}
    t_traj, x_traj = map_pp_to_trajectory(hm, k, pp)
end

######### Define nominal and population simulations

function simulate(m::Simulatable)
    return simulate(m, m.k_nominal)
end
function simulate(m::Simulatable, k::Parameters, N::Int64)
    outputs = collect(zip([simulate(m, k) for i in 1:N]...))
    return collect.(outputs)
end
function simulate(m::Simulatable, N::Int64)
    return simulate(m, m.k_nominal, N)
end

function complete(m::Simulatable, c::Coupling)
    return complete(m, m.k_nominal, c)
end
function complete(m::Simulatable, k::Parameters, coupling_set::Array{<:Coupling,1})
    outputs = collect(zip([complete(m, k, coupling) for coupling in coupling_set]...))
    return collect.(outputs)
end
function complete(m::Simulatable, coupling_set::Array{<:Coupling,1})
    return complete(m, m.k_nominal, coupling_set)
end