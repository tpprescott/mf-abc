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

using Distributions
function tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    
    function num_fired_reactions(x::Array{Float64,1})
        a = propensity_fun(x,k)
        return rand.(Poisson.(a*tau))
    end

    t_traj=Array{Float64,1}()
    x_traj=Array{Array{Float64,1},1}()

    t=0
    x = x0
    push!(t_traj,t)
    push!(x_traj,x)

    while t<T
        t = t+tau
        
        fired_reactions = num_fired_reactions(x)
        x = x+nu*fired_reactions

        push!(t_traj,t)
        push!(x_traj,x)
    end

    return t_traj, x_traj

end

function compare_DM_tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    quick = @time tauleap(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k,tau::Float64)
    slow = @time gillespieDM(nu::Array{Float64,2},propensity_fun::Function,T::Float64,x0::Array{Float64,1},k)
    return quick, slow
end