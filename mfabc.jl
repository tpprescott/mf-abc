struct MFABC
    syn_data::Function       # Mapping nominal parameter to point in summary statistic space
    draw_k::Function    # Draw a parameter
    lofi::Function      # Map parameter to draw (lofi model) in summary statistic space
    hifi::Function      # Map parameter to draw (hifi model) in summary statistic space (accepts output from lofi for coupling)
    dist::Function      # Distance in summary statistic space
end

function runpair(mfabc::MFABC, i::Int64=1; timed_flag::Bool=true)
    k = mfabc.draw_k()

    if timed_flag
    # Simulate low-fidelity and complete high-fidelity
        (yc,pass),ctilde = @timed mfabc.lofi(k)
        (yf),cc = @timed mfabc.hifi(k,pass)
    else
        yc, pass = mfabc.lofi(k)
        yf = mfabc.hifi(k,pass)
    end
    
    # Find out distances from data
    dc = mfabc.dist(yc, mfabc.syn_data())
    df = mfabc.dist(yf, mfabc.syn_data())

    if timed_flag
        return (k..., dc, df, ctilde, cc)
    else
        return (k..., dc, df)
    end
end