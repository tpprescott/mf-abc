using Random
using Distributions

######## Direct simulation

function gillespieDM(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k)
    
    function gillespie_step(x::Array{Float64,1})
        a = propensity_fun(x,k)
        a0 = sum(a)
        aa = cumsum(a)./a0
        
        return log(1/rand())/a0, 1+count(aa.<rand())
    end 

    t_traj=Array{Float64,1}()
    x_traj=Array{Array{Float64,1},1}()
    
    t = 0
    x = x0

    push!(t_traj,t)
    push!(x_traj,x)

    while t<T
        tau_f, j = gillespie_step(x)
        t = t+tau_f
        x = x+nu[:,j]
        push!(t_traj,t)
        push!(x_traj,x)
    end

    return t_traj, x_traj

end

######## Tau Leap Simulation
# Produces trajectory and coarse-grained description of underlying Poisson process

function tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    
    function num_fired_reactions(x::Array{Float64,1})
        a = propensity_fun(x,k)
        return rand.(Poisson.(a*tau)), a*tau
    end

    t_traj=Array{Float64,1}()
    x_traj=Array{Array{Float64,1},1}()
    d_traj=Array{Array{Float64,1},1}()
    f_traj=Array{Array{Int64,1},1}()

    t=0
    x = x0
    push!(t_traj,t)
    push!(x_traj,x)

    while t<T
        t = t+tau
        
        fired_reactions, distance_travelled = num_fired_reactions(x)
        x = x+nu*fired_reactions

        push!(t_traj,t)
        push!(x_traj,x)
        push!(d_traj,distance_travelled)
        push!(f_traj,fired_reactions)
    end

    return t_traj, x_traj, d_traj, f_traj

end

######## Complete a tau-leap into full-scale simulation by
# (A) making the coarse-grained Poisson process fine-grained
# (B) mapping the Poisson process to an exact trajectory

function bridge_PP(d::Array{Float64,1},f::Array{Int64,1})
    D = [0;cumsum(d)]
    pp_fired = Array{Float64,1}()
    for i in 1:length(d)
        if d[i]>0
            append!(pp_fired,rand(Uniform(D[i], D[i+1]), f[i]))
        end
    end
    return sort!(pp_fired)
end

function map_pp_to_trajectory(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,
    pp_set::Array{Array{Float64,1},1})

    t_traj = Array{Float64,1}()
    x_traj = Array{Array{Float64,1},1}()

    t=0
    x = x0
    push!(t_traj,t)
    push!(x_traj,x)

    d = zeros(size(pp_set))
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

        push!(t_traj,t)
        push!(x_traj,x)
    end

    return t_traj, x_traj
end

function complete_tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,
    d_traj::Array{Array{Float64,1},1}, f_traj::Array{Array{Int64,1},1})

    num_r = size(nu,2)
    reaction_d = [[d[i] for d in d_traj] for i in 1:num_r]
    reaction_f = [[f[i] for f in f_traj] for i in 1:num_r]
    bridged_pp = bridge_PP.(reaction_d,reaction_f)

    t_traj, x_traj = map_pp_to_trajectory(nu,propensity_fun,T,x0,k,bridged_pp)
    return t_traj, x_traj
end

######## Timings

function compare_DM_tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    quick = @time tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    slow = @time gillespieDM(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k)
    return quick, slow
end