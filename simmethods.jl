using LinearAlgebra
using Dierckx
using Random
using Distributions
using Printf

mutable struct ModelGenerator
    # Required fields
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity::Function            # Propensity function
    x0::Array{Float64,1}            # Initial conditions
    k_nominal                       # Nominal parameters - type has to match input type of propensity
    T::Float64                      # Simulation time

    # Optional fields
    diff_propensity::Function       # Differentiated propensity function (for adaptivity of tau-leaping)
    stochastic_reactions::BitArray  # True/False whether the reactions are stochastic
    deterministic_step::Function    # The deterministic reaction wait time and result

    function ModelGenerator(nu::Matrix{Float64}, propensity::Function, x0::Array{Float64,1}, k_nominal, T::Float64)
        mg = new()
        mg.nu = nu
        mg.propensity = propensity
        mg.x0 = x0
        mg.k_nominal = k_nominal
        mg.T = T
        return mg
    end 

end


######## Direct simulation

struct GillespieModel
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity::Function            # Propensity function
    x0::Array{Float64,1}            # Initial conditions
    k                               # Parameters to simulate - type has to match input type of par_prior
    T::Float64                      # Simulation time
    
    GillespieModel(nu,propensity,x0,k,T) = new(nu,propensity,x0,k,T)
    function GillespieModel(mg::ModelGenerator) 
        return new(mg.nu, mg.propensity, mg.x0, mg.k_nominal, mg.T)
    end
    function GillespieModel(mg::ModelGenerator, parameters)
        return new(mg.nu, mg.propensity, mg.x0, parameters, mg.T)
    end
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

function propensity!(speeds::Array{Float64,1}, x::Array{Float64,1}, model)
    speeds[:] = model.propensity(x, model.k)
    return nothing
end

function gillespieDM(gm::GillespieModel)

    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(gm.x0)
    x::Array{Float64,1} = copy(gm.x0)
    speeds::Array{Float64,1} = gm.propensity(gm.x0, gm.k)

    while t_traj[end] < gm.T
        propensity!(speeds, x, gm)
        gillespie_update!(t_traj, x, speeds, gm)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    (nx,nr) = size(gm.nu)

    return t_traj, reshape(x_traj,(nx, nt))

end

######## Tau Leap Simulation
# Produces trajectory and coarse-grained description of underlying Poisson process
# The tau leap size adapts according to epsilon (if positive) to limit the rate of change
# of the propensity function
# The parameter nc defines reactions as critical if nc firings would send molecule 
# counts negative: then the reactions are not allowed to proceed according to tau leap.

struct TauLeapModel
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity::Function            # Propensity function
    diff_propensity::Function       # Differentiated propensity function
    x0::Array{Float64,1}            # Initial conditions
    k                               # Parameters to simulate - type has to match input type of propensity
    T::Float64                      # Simulation time
    
    TauLeapModel(nu,propensity,diff_propensity,x0,k,T,summ_stat) = new(nu,propensity,diff_propensity,x0,k,T)
    function TauLeapModel(mg::ModelGenerator)
        return new(mg.nu, mg.propensity, mg.diff_propensity, mg.x0, mg.k_nominal, mg.T)
    end
    function TauLeapModel(mg::ModelGenerator, parameters)
        return new(mg.nu, mg.propensity, mg.diff_propensity, mg.x0, parameters, mg.T)
    end
end

function diff_propensity!(diffspeeds::Array{Float64,2}, x::Array{Float64, 1}, tlm::TauLeapModel)
    diffspeeds[:,:] = tlm.diff_propensity(x, tlm.k)
    return nothing
end

function tauleap(tlm::TauLeapModel; tau::Float64=0.01, nc::Float64=0.0, epsilon::Float64=0.0)
    
    function decide_critical!(critical::Array{Bool,1}, x::Array{Float64,1}, speeds::Array{Float64,1}, L::Array{Float64,1}, critnu::Array{Bool,2}, posscrit::Array{Bool,1}, nc::Float64, tlm::TauLeapModel)
        for j in 1:length(critical)
            if posscrit[j]
                if speeds[j]==0
                    critical[j] = false
                else
                    L[j] = minimum(ceil.(x[critnu[:,j]]./abs.(tlm.nu[critnu[:,j],j])))
                    critical[j] = (L[j] <= nc)
                end
            end
        end
        return nothing
    end

    function largest_timestep(speeds::Array{Float64,1}, diffspeeds::Array{Float64,2}, critical::Array{Bool,1}, epsilon::Float64, tlm::TauLeapModel)
        # largest_timestep(speeds, diffspeeds, critical, epsilon, tlm)
        
        nr = size(tlm.nu,2)
        
        if .&(critical...)
            return Inf64
        else
            F = diffspeeds[.!critical, :] * tlm.nu[:, .!critical]
            mu = F*speeds[.!critical]
            sigma2 = (F.^2)*speeds[.!critical]
            v1 = epsilon*sum(speeds)./abs.(mu)
            v2 = ((epsilon*sum(speeds))^2)./sigma2
            return minimum([v1;v2])
        end
    end

    function do_tau_leap!(t_traj, x, d, f, tauprime, tauprimeprime, speeds, critical, tlm)
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
    d = zeros(nr)
    f = zeros(nr)
    
    L::Array{Float64,1} = zeros(nr)
    speeds::Array{Float64,1} = zeros(nr)
    diffspeeds::Array{Float64,2} = zeros(nr, nx)

    critnu::Array{Bool,2} = (tlm.nu .< 0)
    critical::Array{Bool,1} = [minimum(tlm.nu[:,j])<0 for j in 1:nr]
    posscrit::Array{Bool,1} = [minimum(tlm.nu[:,j])<0 for j in 1:nr]
    
    
    while t_traj[end] < tlm.T
        propensity!(speeds, x, tlm)
        decide_critical!(critical, x, speeds, L, critnu, posscrit, nc, tlm)

        if epsilon>0
            diff_propensity!(diffspeeds, x, tlm)
            tauprime = largest_timestep(speeds, diffspeeds, critical, epsilon, tlm)
        else
            tauprime = tau
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
    return t_traj, reshape(x_traj,(nx, nt)), reshape(d_traj,(nr, nt)), reshape(f_traj,(nr, nt))

end

######## Hybrid deterministic/continuous simulation
# Produces trajectory and Poisson process of the stochastic firings

struct HybridModel
    nu::Matrix{Float64}             # Stoichiometric matrix
    propensity::Function            # Propensity function
    x0::Array{Float64,1}            # Initial conditions
    k                               # Parameters to simulate - type has to match input type of par_prior
    T::Float64                      # Simulation time
    stochastic_reactions::BitArray  # True/False whether the reactions are stochastic
    deterministic_step::Function    # The deterministic reaction wait time and result
    
    HybridModel(nu,propensity,x0,k,T) = new(nu,propensity,x0,k,T,stochastic_reactions,deterministic_step)
    function HybridModel(mg::ModelGenerator) 
        return new(mg.nu, mg.propensity, mg.x0, mg.k_nominal, mg.T, mg.stochastic_reactions, mg.deterministic_step)
    end
    function HybridModel(mg::ModelGenerator, parameters)
        return new(mg.nu, mg.propensity, mg.x0, parameters, mg.T, mg.stochastic_reactions, mg.deterministic_step)
    end
end

function reduced_propensity!(speeds::Array{Float64,1}, x::Array{Float64,1}, model::HybridModel)
    speeds[:] = model.propensity(x, model.k) .* model.stochastic_reactions
    return nothing
end

function hybrid_step!(t_traj::Array{Float64,1}, x::Array{Float64,1}, d::Array{Float64,1}, pp_set::Array{Array{Float64,1},1}, speeds::Array{Float64,1}, hm::HybridModel)
    
    tau_det, nu_j_det = hm.deterministic_step(t_traj[end], x, hm.k)    
    a0 = sum(speeds)
    if a0>0
        tau_sto = rand(Exponential(1/a0))
    else
        tau_sto = Inf64
    end

    if tau_det <= tau_sto
        append!(t_traj, t_traj[end]+tau_det)
        x[:] += nu_j_det
        d[:] += tau_det*speeds
    else
        append!(t_traj, t_traj[end]+tau_sto)
        j = rand(Categorical(speeds./a0))
        x[:] += hm.nu[:,j]
        d[:] += tau_sto*speeds
        append!(pp_set[j],d[j])
    end
end

function hybrid(hm::HybridModel)

    (n_x, n_r) = size(hm.nu)

    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(hm.x0)
    x::Array{Float64,1} = copy(hm.x0)
    
    pp_set=[Float64[] for i in 1:n_r]
    d::Array{Float64,1} = zeros(n_r)
    speeds::Array{Float64,1} = zeros(n_r)
    
    while t_traj[end] < hm.T
        reduced_propensity!(speeds, x, hm)
        hybrid_step!(t_traj, x, d, pp_set, speeds, hm)
        append!(x_traj,x)
    end

    n_t = length(t_traj)
    return t_traj, reshape(x_traj,(n_x, n_t)), pp_set
end


######## Coupling methods, based on
# (A) making a coarse-grained Poisson process fine-grained
# (B) mapping a known Poisson process to an exact trajectory (filling in gaps where needed)

function bridge_PP(d::Array{Float64,2},f::Array{Int64,2})
    nr = size(d,1)
    firings = d[:,2:end].*rand.(f[:,2:end])
    D = cumsum(d,dims=2)
    pp_split = broadcast((a,b)->a.+b, D[:,1:end-1], firings)
    sort!.(pp_split)
    pp_fired = [vcat(pp_split[j,:]...) for j = 1:nr]
    return pp_fired
end

function gillespie_update!(t_traj::Array{Float64,1}, x::Array{Float64,1}, next_event_d::Array{Float64, 1}, d::Array{Float64,1}, pps::Array{Array{Float64,1},1}, a::Array{Float64,1}, mdl)
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

function map_pp_to_trajectory(mdl::Union{TauLeapModel,HybridModel}, pp_set::Array{Array{Float64,1},1})

    t_traj::Array{Float64,1} = [0.0]
    x_traj::Array{Float64,1} = copy(mdl.x0)
    x = copy(mdl.x0)
    pp = deepcopy(pp_set)
    
    (nx,nr) = size(mdl.nu)
    d = zeros(nr)
    speed = zeros(nr)

    next_event_d = randexp(nr)
    for j in 1:nr
        if ~isempty(pp[j])
            next_event_d[j] = popfirst!(pp[j])
        end
    end

    while t_traj[end] < mdl.T
        propensity!(speed, x, mdl)
        gillespie_update!(t_traj, x, next_event_d, d, pp, speed, mdl)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx,nt))
end


######### Apply coupling methods

function complete_tauleap(tlm::TauLeapModel, d_traj::Array{Float64,2}, f_traj::Array{Int64,2})
    t_traj, x_traj = map_pp_to_trajectory(tlm,bridge_PP(d_traj, f_traj))
end

function complete_hybrid(hm::HybridModel, pp_set::Array{Array{Float64,1},1})
    t_traj, x_traj = map_pp_to_trajectory(hm, pp_set)
end