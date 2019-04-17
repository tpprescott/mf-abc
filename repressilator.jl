include("simmethods.jl")
include("mfabc.jl")

function repressilator_model_generator()
    # Require nu, propensity, x0, ko, T, parameter_uncertainty
    # Optional: diff_propensity

    z =  zeros(3,3)
    nu = vcat(hcat(I,-I,z,z),hcat(z,z,I,-I))
    x0 = [0.0; 0.0; 0.0; 40.0; 20.0; 60.0]
    ko = (1.0, 2.0, 5.0, 1000.0, 20.0)
    T = 10.0

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

    function diff_propensity(x::Array{Float64,1}, k::NTuple{5,Float64})

        dp = zeros(12,6);

        dp[1,6] = -k[4]*k[2].*(k[5].^k[2]).*(x[6].^(k[2]-1))*(k[5].^k[2] + x[6].^k[2]).^(-2);
        dp[2,4] = -k[4]*k[2].*(k[5].^k[2]).*(x[4].^(k[2]-1))*(k[5].^k[2] + x[4].^k[2]).^(-2);
        dp[3,5] = -k[4]*k[2].*(k[5].^k[2]).*(x[5].^(k[2]-1))*(k[5].^k[2] + x[5].^k[2]).^(-2);
        dp[4:6,1:3] = Matrix{Float64}(I,3,3);
        dp[7:9,1:3] = k[3]*Matrix{Float64}(I,3,3);
        dp[10:12,4:6] = k[3]*Matrix{Float64}(I,3,3);

        return dp 
    end

    
    function parameter_uncertainty(nominal_parameters::NTuple{5,Float64})
        k1 = nominal_parameters[1]
        k2 = rand(Uniform(1,4))
        k3 = nominal_parameters[3]
        k4 = nominal_parameters[4]
        k5 = rand(Uniform(10,30))
        return k1,k2,k3,k4,k5
    end


    repressilator = ModelGenerator(nu, propensity, x0, ko, T, parameter_uncertainty)
    repressilator.diff_propensity = diff_propensity
    return repressilator

end

function repressilator_mfabc_problem()
    rep_mg = repressilator_model_generator()
    
    function summary_statistics(t::Array{Float64,1}, x::Array{Float64,2})
        return vcat([Spline1D(t,x[j,:];k=1)(0:10) for j in 1:size(x,1)]...)
    end

    function syn_data()
        Random.seed!(123)
        t,x = gillespieDM(GillespieModel(rep_mg, nominal=true))
        Random.seed!()
        return summary_statistics(t,x)
    end

    draw_k() = rep_mg.par_uncertainty(rep_mg.k_nominal)

    function lofi(k)
        tlm = TauLeapModel(rep_mg,k)
        t,x,d,f = tauleap(tlm, tau=0.01, nc=3.0, epsilon=0.01)
        return summary_statistics(t,x), (tlm,d,f)
    end

    function hifi(k,pass)
        t,x = complete_tauleap(pass...)
        return summary_statistics(t,x)
    end

    function dist(y1::Array{Float64,1},y2::Array{Float64,1})
        return norm(y2-y1)/rep_mg.T
    end
    
    return MFABC(syn_data, draw_k, lofi, hifi, dist)
end

function viral_model_generator()
    # Require nu, propensity, x0, ko, T, parameter_uncertainty
    # Optional: diff_propensity

    nu = Float64.([0 1 0 -1 0 0 ; 1 -1 0 0 0 -1 ; 0 0 1 0 -1 -1; 0 0 0 0 0 1])

    function propensity(x::Array{Float64,1}, k::NTuple{6,Float64})
        return k.*[x[1], x[2], x[1], x[1], x[3], x[2]*x[3]]
    end

    x0 = [1.0, 0.0, 0.0, 0.0]
    ko = (1.0, 0.025, 100.0, 0.25, 1.9985, 7.5e-5)
    T = 200.0

    function parameter_uncertainty(nominal_parameters::NTuple{6,Float64})
        k1 = nominal_parameters[1]*(1.5^rand(Uniform(-1,1))) # Scale by factor between 2/3 and 3/2
        k2,k3,k4,k5,k6 = nominal_parameters[2:6]
        return k1,k2,k3,k4,k5,k6
    end

    viral = ModelGenerator(nu, propensity, x0, ko, T, parameter_uncertainty)

    stochastic_reactions = [true,true,false,true,false,true]

    function deterministic_step(t,x,k)
        err = (k[3]/k[5])*x[1] - x[3]
        if abs(1/err)<1
            tau = (-1/k[5]) * log(1 - abs(1/err))
        else
            tau = T-t
            err = 0
        end
        return tau, [0, 0, sign(err), 0]
    end

    viral.stochastic_reactions = stochastic_reactions
    viral.deterministic_step = deterministic_step
    return viral

end