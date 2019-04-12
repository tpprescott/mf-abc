include("repressilator.jl")
include("simmethods.jl")

Random.seed!(123)
# Create observed data using nominal parameters
to,xo = gillespieDM(nu, propensity, T, x0, ko)
yo = summary_statistics(to,xo)
Random.seed!()

function runpair(i::Int64=1; 
    timed_flag::Bool=true, epsilon::Float64=0.05, nc::Float64=5.0)

    # Choose a parameter
    k = prior_sample()

    if timed_flag
    # Simulate lofidelity and complete hifidelity
        (tc,xc,d,f),ctilde = @timed tauleap(nu,propensity,diff_propensity,T,x0,k,epsilon=epsilon,nc=nc)
        (tf,xf),cc = @timed complete_tauleap(nu,propensity,T,x0,k,d,f)
    else
        tc,xc,d,f = tauleap(nu,propensity,T,x0,k,epsilon=epsilon,nc=nc)
        tf,xf = complete_tauleap(nu,propensity,T,x0,k,d,f)
    end
    
    # Find out distances from data
    dc = ss_distance(summary_statistics(tc,xc),yo,T)
    df = ss_distance(summary_statistics(tf,xf),yo,T)

    if timed_flag
        return (k..., dc, df, ctilde, cc)
    else
        return (k..., dc, df)
    end
end