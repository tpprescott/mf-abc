include("repressilator.jl")
include("simmethods.jl")

Random.seed!(123)
# Create observed data using nominal parameters
to,xo = gillespieDM(nu, propensity, T, x0, ko)
yo = summary_statistics(to,xo)

function runpair()
    # Choose a parameter
    k = prior_sample()

    # Simulate lofidelity and complete hifidelity
    (tc,xc,d,f),ctilde = @timed tauleap(nu,propensity,T,x0,k,0.01)
    (tf,xf),cc = @timed complete_tauleap(nu,propensity,T,x0,k,d,f)
    
    # Find out distances from data
    dc = ss_distance(summary_statistics(tc,xc),yo)
    df = ss_distance(summary_statistics(tf,xf),yo)

    return k, ctilde, cc, dc, df
end