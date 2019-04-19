include("simmethods.jl")
include("mfabc.jl")

# Repressilator model specification contains:
# - Model in repressilator_model_generator - what to simulate
# - mf-abc specification (what is low-fidelity, high-fidelity, etc) in repressilator_mfabc_problem
# - Prior parameter distribution in repressilator_prior

function repressilator_model_generator()
    # Require nu, propensity, x0, ko, T, parameter_uncertainty
    # Optional: diff_propensity

    z =  zeros(3,3)
    nu = vcat(hcat(I,-I,z,z),hcat(z,z,I,-I))
    x0 = [0.0; 0.0; 0.0; 40.0; 20.0; 60.0]
    ko = (1.0, 2.0, 5.0, 1000.0, 20.0)
    T = 10.0

    function propensity!(v::Array{Float64,1}, x::Array{Float64,1}, k::NTuple{5,Float64})
                
        v[1:3] = k[1] .+ (k[4]*(k[5]^k[2]))./((k[5]^k[2]) .+ (x[[6,4,5]].^k[2]))
        v[4:6] = x[1:3]
        v[7:9] = k[3].*x[1:3]
        v[10:12] = k[3].*x[4:6]
        
        return nothing
    end

    function diff_propensity!(dv::Array{Float64,2}, x::Array{Float64,1}, k::NTuple{5,Float64})

        dv[1,6] = -k[4]*k[2].*(k[5].^k[2]).*(x[6].^(k[2]-1))*(k[5].^k[2] + x[6].^k[2]).^(-2);
        dv[2,4] = -k[4]*k[2].*(k[5].^k[2]).*(x[4].^(k[2]-1))*(k[5].^k[2] + x[4].^k[2]).^(-2);
        dv[3,5] = -k[4]*k[2].*(k[5].^k[2]).*(x[5].^(k[2]-1))*(k[5].^k[2] + x[5].^k[2]).^(-2);
        dv[4:6,1:3] = Matrix{Float64}(I,3,3);
        dv[7:9,1:3] = k[3]*Matrix{Float64}(I,3,3);
        dv[10:12,4:6] = k[3]*Matrix{Float64}(I,3,3);

        return nothing 
    end

    repressilator = ModelGenerator(nu, propensity!, x0, ko, T)
    repressilator.diff_propensity! = diff_propensity!
    return repressilator

end

function repressilator_mfabc_problem(parameter_sampler::Function)

    rep_mg = repressilator_model_generator()
    
    function summary_statistics(t::Array{Float64,1}, x::Array{Float64,2})
        return vcat([Spline1D(t,x[j,:];k=1)(0:rep_mg.T) for j in 1:size(x,1)]...)
    end

    function lofi(k)
        tlm = TauLeapModel(rep_mg, k)
        t,x,d,f = tauleap(tlm, tau=0.01, nc=3.0, epsilon=0.01)
        return summary_statistics(t,x), (tlm,d,f)
    end

    # The following hifi *couples* low fidelity and high fidelity simulations:
    function hifi(k, pass)
        t, x = complete_tauleap(pass...)
        return summary_statistics(t,x)
    end

    # # This hifi version would produce independent (uncoupled) high fidelity simulations:
    # function hifi(k,pass)
    #     gm = GillespieModel(rep_mg, k)
    #     t, x = gillespieDM(gm)
    #     return summary_statistics(t,x)
    # end

    Random.seed!(123)
    t,x = gillespieDM(GillespieModel(rep_mg)) # Simulate the nominal model with a fixed seed
    Random.seed!()
    yo = summary_statistics(t,x)

    function distance(y::Array{Float64,1})
        return norm(y-yo)/rep_mg.T
    end
    
    return MFABC(parameter_sampler, lofi, hifi, distance)
end

function repressilator_prior()

    ko = repressilator_model_generator().k_nominal

    k2 = rand(Uniform(1,4))
    k5 = rand(Uniform(10,30))

    return ko[1], k2, ko[3], ko[4], k5
end