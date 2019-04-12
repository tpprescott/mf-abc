using Random
using Distributions

######## Direct simulation

function gillespieDM(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k)
    
    function gillespie_step(x::Array{Float64,1})
        a = propensity_fun(x,k)
        a0 = sum(a)
        aa = a./a0
        
        return rand(Exponential(1/a0)), rand(Categorical(aa))
    end 

    t_traj=Array{Float64,1}()
    x_traj=Array{Float64,1}()
    
    t = 0
    x = x0

    append!(t_traj,t)
    append!(x_traj,x)

    while t<T
        tau_f, j = gillespie_step(x)
        t = t+tau_f
        x = x+nu[:,j]
        append!(t_traj,t)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    (nx,nr) = size(nu)

    return t_traj, reshape(x_traj,(nx, nt))

end

######## Tau Leap Simulation
# Produces trajectory and coarse-grained description of underlying Poisson process
# The tau leap size adapts according to epsilon (if positive) to limit the rate of change
# of the propensity function
# The parameter nc defines reactions as critical if nc firings would send molecule 
# counts negative: then the reactions are not allowed to proceed according to tau leap.

function tauleap(nu::Array{Float64,2},propensity_fun::Function,diff_propensity_fun::Function,T::Float64,x0::Array{Float64,1},k;
    tau::Float64=0.01,nc::Float64=0.0,epsilon::Float64=0.0)
    
    function decide_critical(nu_j::Array{Float64,1},x::Array{Float64,1},speed::Float64,nc::Float64)
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
        try_x = x + nu*fired_reactions
    
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

    nx, nr = size(nu)
    if nc==0
        posscrit = []
    else
        posscrit = filter(j->(minimum(nu[:,j])<0),1:nr)
    end
    notcrit = setdiff(1:nr,posscrit)
    
    t=0
    x = x0
    append!(t_traj,t)
    append!(x_traj,x)
    append!(d_traj,zeros(nr))
    append!(f_traj,zeros(nr))

    while t<T
        speeds = propensity_fun(x,k)
        crit_j = filter(j->decide_critical(nu[:,j],x,speeds[j],nc),posscrit)
        notcrit_j = setdiff(1:nr,crit_j)

        if epsilon>0
            diff_speeds = diff_propensity_fun(x,k)
            tauprime = largest_timestep(nu, speeds, diff_speeds, notcrit_j, epsilon)
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

function map_pp_to_trajectory(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,
    pp_set::Array{Array{Float64,1},1})

    t_traj = Array{Float64,1}()
    x_traj = Array{Float64,1}()

    t=0
    x = x0
    
    append!(t_traj,t)
    append!(x_traj,x)

    (nx,nr) = size(nu)
    d = zeros(nr)
    next_event_d = [popfirst!(pp) for pp in pp_set]

    while t<T
        speed = propensity_fun(x,k)
        (tau,j) = findmin((next_event_d - d)./speed)
        d += speed*tau
        next_event_d[j] = try 
            popfirst!(pp_set[j])
        catch
            next_event_d[j] + randexp()
        end
        
        t += tau
        x += nu[:,j]

        append!(t_traj,t)
        append!(x_traj,x)
    end

    nt = length(t_traj)
    return t_traj, reshape(x_traj,(nx,nt))
end

function complete_tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,
    d_traj::Array{Float64,2}, f_traj::Array{Int64,2})

    t_traj, x_traj = map_pp_to_trajectory(nu,propensity_fun,T,x0,k,bridge_PP(d_traj,f_traj))

end

######## Timings

function compare_DM_tauleap(nu::Array{Float64,2},propensity_fun::Function,diff_propensity_fun::Function,T::Float64,x0::Array{Float64,1},k
    ; nc::Float64, epsilon::Float64, tau::Float64)

    (tc,xc,d,f),ctilde = @timed tauleap(nu,propensity_fun,diff_propensity_fun,T,x0,k, epsilon=epsilon, nc=nc, tau=tau)
    (tf,xf),c_c = @timed complete_tauleap(nu,propensity,T,x0,k,d,f)
    (t,x),c = @timed gillespieDM(nu,propensity_fun,T,x0,k)
    return ctilde, c_c, c
end