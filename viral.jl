include("simmethods.jl")
include("mfabc.jl")

# Viral model specification contains:
# - Model in viral_model_generator - what to simulate
# - mf-abc specification (what is low-fidelity, high-fidelity, etc) in viral_mfabc_problem
# - Prior parameter distribution in viral_prior

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

    viral = ModelGenerator(nu, propensity, x0, ko, T)

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

function viral_mfabc_problem(parameter_sampler::Function)

    vir_mg = viral_model_generator()
    
    function summary_statistics(t_pop::Array{Array{Float64,1},1}, x_pop::Array{Array{Float64,2},1})
        return sort([x[end,end] for x in x_pop])
    end

    function simulate_population(sim_method, model, input::Int64=10)
        outputs = collect(zip([sim_method(model) for i in 1:input]...))
        return collect.(outputs)
    end
    function simulate_population(sim_method, model, input::AbstractArray)
        outputs = collect(zip([sim_method(model, coupling) for coupling in input]...))
        return collect.(outputs)
    end

    function lofi(k)
        hm = HybridModel(vir_mg, k)
        t_pop, x_pop, pp_set_pop = simulate_population(hybrid, hm)
        return summary_statistics(t_pop, x_pop), (hm, pp_set_pop)
    end

    # The following hifi *couples* low fidelity and high fidelity simulations:
    function hifi(k, pass)
        t_pop, x_pop = simulate_population(complete_hybrid, pass[1], pass[2])
        return summary_statistics(t_pop, x_pop)
    end

    # # This hifi version would produce independent (uncoupled) high fidelity simulations:
    # function hifi(k, pass)
    #     gm = GillespieModel(vir_mg, k)
    #     t, x = gillespieDM(gm)
    #     return summary_statistics(t,x)
    # end

    Random.seed!(123)
    t_pop, x_pop = simulate_population(gillespieDM, GillespieModel(vir_mg)) # Simulate the nominal model with a fixed seed
    Random.seed!()
    yo = summary_statistics(t_pop, x_pop)

    function distance(y::Array{Float64,1})
        return norm(y-yo)/vir_mg.T
    end
    
    return MFABC(parameter_sampler, lofi, hifi, distance)
end

function viral_prior()

    ko = viral_model_generator().k_nominal
    k1 = ko[1] * 1.5^rand(Uniform(-1,1)) # Scale first parameter by factor between 2/3 and 3/2

    return k1, ko[2], ko[3], ko[4], ko[5], ko[6]

end