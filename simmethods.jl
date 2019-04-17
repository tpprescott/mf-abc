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
    k_nominal                       # Nominal parameters - type has to match input type of propensity and par_prior
    T::Float64                      # Simulation time
    par_uncertainty::Function       # Generate uncertain parameters

    # Optional fields
    diff_propensity::Function       # Differentiated propensity function (for adaptivity of tau-leaping)
    stochastic_reactions::BitArray  # True/False whether the reactions are stochastic
    deterministic_step::Function    # The deterministic reaction wait time and result

    function ModelGenerator(nu::Matrix{Float64}, propensity::Function, x0::Array{Float64,1}, k_nominal, T::Float64, par_uncertainty::Function)
        mg = new()
        mg.nu = nu
        mg.propensity = propensity
        mg.x0 = x0
        mg.k_nominal = k_nominal
        mg.T = T
        mg.par_uncertainty = par_uncertainty
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
    function GillespieModel(mg::ModelGenerator; nominal::Bool=false) 
        if nominal
            k = mg.k_nominal
        else
            k = mg.par_uncertainty(mg.k_nominal)
        end
        return new(mg.nu, mg.propensity, mg.x0, k, mg.T)
    end
    function GillespieModel(mg::ModelGenerator, k_override)
        return new(mg.nu, mg.propensity, mg.x0, k_override, mg.T)
    end
end

function gillespieDM(gm::GillespieModel)
    
    function gillespie_update(t::Float64, x::Array{Float64,1})
        a = gm.propensity(x,gm.k)
        a0 = sum(a)
        if a0>0
            aa = a./a0
            t += rand(Exponential(1/a0))
            x += gm.nu[:, rand(Categorical(aa))]
        else
            print("Propensities: "*prod([@sprintf("%.3f ; ",ai) for ai in a]))
            t = gm.T
        end
        return t, x
    end 

    t_traj=Array{Float64,1}()
    x_traj=Array{Float64,1}()
    
    t = 0.0
    x = gm.x0

    append!(t_traj,t)
    append!(x_traj,x)

    while t<gm.T
        t, x = gillespie_update(t, x)
        append!(t_traj,t)
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
    function TauLeapModel(mg::ModelGenerator; nominal::Bool=false)
        if nominal
            k = mg.k_nominal
        else
            k = mg.par_uncertainty(mg.k_nominal)
        end
        return new(mg.nu, mg.propensity, mg.diff_propensity, mg.x0, k, mg.T)
    end
    function TauLeapModel(mg::ModelGenerator, k_override)
        return new(mg.nu, mg.propensity, mg.diff_propensity, mg.x0, k_override, mg.T)
    end
end

function tauleap(tlm::TauLeapModel; tau::Float64=0.01, nc::Float64=0.0, epsilon::Float64=0.0)
    
    function decide_critical(nu_j::Array{Float64,1}, x::Array{Float64,1}, speed::Float64, nc::Float64)
        if speed==0
            return false
        end
        L_j = minimum([ceil(x_i/abs(nu_ij)) for (x_i, nu_ij) in zip(x, nu_j) if nu_ij<0])
        return L_j <= nc
    end

    function largest_timestep(nu::Array{Float64,2}, speeds::Array{Float64,1}, diff_speeds::Array{Float64,2}, notcrit::Array{Int64,1},
        epsilon::Float64)
        
        nr = size(nu,2)
        
        if isempty(notcrit)
            return Inf64
        else
            F = diff_speeds[notcrit,:] * nu[:,notcrit]
            mu = F*speeds[notcrit]
            sigma2 = (F.^2)*speeds[notcrit]
            v1 = epsilon*sum(speeds)./abs.(mu)
            v2 = ((epsilon*sum(speeds))^2)./sigma2
            return minimum([v1;v2])
        end
    end

    function get_step(tauprime, tauprimeprime, t, x, speeds, crit, noncrit)
        
        nr = length(speeds)

        tau_adaptive = min(tauprime,tauprimeprime)
        distance_travelled = tau_adaptive*speeds

        fired_reactions = zeros(nr)
        fired_reactions[noncrit] .= rand.(Poisson.(distance_travelled[noncrit]))
        # Only looking at non-critical fired reactions

        if tauprimeprime < tauprime
            crit_k = rand(Categorical((speeds[crit])/sum(speeds[crit])))
            fired_reactions[crit[crit_k]] = 1 # Add in a single critical reaction if it fired first
        end

        try_t = t + tau_adaptive
        try_x = x + tlm.nu*fired_reactions
    
        if .|((try_x.<0)...)
            # print("Halving!\n")
            return t,x,d,f = get_step(tauprime/2, tauprimeprime, t, x, speeds, crit, notcrit) 
            # i.e. if any x component is negative, half tauprime and go again
        end

        return try_t, try_x, distance_travelled, fired_reactions
    end

    t_traj=Array{Float64,1}()
    x_traj=Array{Float64,1}()
    d_traj=Array{Float64,1}()
    f_traj=Array{Int64,1}()

    nx, nr = size(tlm.nu)
    if nc==0
        posscrit = []
    else
        posscrit = filter(j->(minimum(tlm.nu[:,j])<0),1:nr)
    end
    notcrit = setdiff(1:nr,posscrit)
    
    t=0
    x = tlm.x0
    append!(t_traj,t)
    append!(x_traj,x)
    append!(d_traj,zeros(nr))
    append!(f_traj,zeros(nr))

    while t<tlm.T
        speeds = tlm.propensity(x,tlm.k)
        crit_j = filter(j->decide_critical(tlm.nu[:,j],x,speeds[j],nc),posscrit)
        notcrit_j = setdiff(1:nr,crit_j)

        if epsilon>0
            diff_speeds = tlm.diff_propensity(x,tlm.k)
            tauprime = largest_timestep(tlm.nu, speeds, diff_speeds, notcrit_j, epsilon)
        else
            tauprime = tau
        end

        if isempty(crit_j)
            tauprimeprime = Inf64
        else
            tauprimeprime = rand(Exponential(1/sum(speeds[crit_j])))
        end

        t, x, d, f = get_step(tauprime, tauprimeprime, t, x, speeds, crit_j, notcrit_j)

        append!(t_traj,t)
        append!(x_traj,x)
        append!(d_traj,d)
        append!(f_traj,f)

    end
    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx, nt)), reshape(d_traj,(nr, nt)), reshape(f_traj,(nr, nt))

end

######## Complete a tau-leap into full-scale simulation by
# (A) making the coarse-grained Poisson process fine-grained
# (B) mapping the Poisson process to an exact trajectory

function bridge_PP(d::Array{Float64,2},f::Array{Int64,2})
    nr = size(d,1)
    firings = d[:,2:end].*rand.(f[:,2:end])
    D = cumsum(d,dims=2)
    pp_split = broadcast((a,b)->a.+b, D[:,1:end-1], firings)
    sort!.(pp_split)
    pp_fired = [vcat(pp_split[j,:]...) for j = 1:nr]
    return pp_fired
end

function map_pp_to_trajectory(tlm::Union{TauLeapModel,HybridModel}, pp_set::Array{Array{Float64,1},1})

    t_traj = Array{Float64,1}()
    x_traj = Array{Float64,1}()

    t=0
    x = tlm.x0
    
    append!(t_traj,t)
    append!(x_traj,x)

    (nx,nr) = size(tlm.nu)
    d = zeros(nr)
    next_event_d = randexp(nr)
    for j in 1:nr
        if ~isempty(pp_set[j])
            next_event_d[j] = popfirst!(pp_set[j])
        end
    end

    while t<tlm.T
        speed = tlm.propensity(x,tlm.k)
        (tau,j) = findmin((next_event_d - d)./speed)
        d += speed*tau
        if isempty(pp_set[j])
            next_event_d[j] += randexp()
        else
            next_event_d[j] = popfirst!(pp_set[j])
        end
        
        t += tau
        x += tlm.nu[:,j]

        append!(t_traj,t)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx,nt))
end

function complete_tauleap(tlm::TauLeapModel, d_traj::Array{Float64,2}, f_traj::Array{Int64,2})
    t_traj, x_traj = map_pp_to_trajectory(tlm,bridge_PP(d_traj,f_traj))
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
    function HybridModel(mg::ModelGenerator; nominal::Bool=false) 
        if nominal
            k = mg.k_nominal
        else
            k = mg.par_uncertainty(mg.k_nominal)
        end
        return new(mg.nu, mg.propensity, mg.x0, k, mg.T, mg.stochastic_reactions, mg.deterministic_step)
    end
    function GillespieModel(mg::ModelGenerator, k_override)
        return new(mg.nu, mg.propensity, mg.x0, k_override, mg.T, mg.stochastic_reactions, mg.deterministic_step)
    end
end

function hybrid(hm::HybridModel)

    (n_x, n_r) = size(hm.nu)

    t_traj=Array{Float64,1}()
    x_traj=Array{Float64,1}()
    
    pp_set=[Float64[] for i in 1:n_r]
    d = zeros(n_r)
    
    t = 0.0
    x = hm.x0
    
    append!(t_traj,t)
    append!(x_traj,x)

    while t<hm.T
        
        tau_det, nu_j_det = hm.deterministic_step(t,x,hm.k)
        speeds = hm.propensity(x,hm.k) .* hm.stochastic_reactions
        
        a0 = sum(speeds)
        if a0>0
            tau_sto = rand(Exponential(1/a0))
        else
            tau_sto = Inf64
        end

        if tau_det <= tau_sto
            t += tau_det
            x += nu_j_det
            d += tau_det*speeds
        else
            t += tau_sto
            j = rand(Categorical(speeds./a0))
            x += hm.nu[:,j]
            d += tau_sto*speeds
            append!(pp_set[j],d[j])
        end

        append!(t_traj,t)
        append!(x_traj,x)
    end

    n_t = length(t_traj)
    return t_traj, reshape(x_traj,(n_x, n_t)), pp_set
end

function complete_hybrid(hm::HybridModel, pp_set::Array{Array{Float64,1},1})
    t_traj, x_traj = map_pp_to_trajectory(hm,pp_set)
end