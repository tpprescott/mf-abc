## Output spaces
export RawTimeSeries, RotatedTimeSeries, Displacements
struct RawTimeSeries <: AbstractSummaryStatisticSpace
    y::Array{Complex{Float64},2}
    t::Array{Float64,1}
    function RawTimeSeries(y::AbstractArray{Complex{Float64}, 2}, t)
        return new(y, t)
    end
end
struct RotatedTimeSeries <: AbstractSummaryStatisticSpace
    y::Array{Complex{Float64},2}
    t::Array{Float64,1}
    function RotatedTimeSeries(y::AbstractArray{Complex{Float64}, 2}, t)
        y_processed = copy(y)
        n = size(y_processed,2)
        for j=1:n
            r_end_j = abs(y_processed[end,j])
            y_processed[:,j] ./= y_processed[end,j]
            y_processed[:,j] .*= r_end_j
        end
        return new(y_processed, t)
    end
end
struct Displacements <: AbstractSummaryStatisticSpace
    NetDisplacement::Array{Float64,1}
    MeanDisplacement::Array{Float64,1}
    StdDisplacement::Array{Float64,1}
    function Displacements(y::AbstractArray{Complex{Float64}, 2}, t)
        n = size(y, 2)
        displacements = diff(y, dims=1)
        
        NetDisplacement = abs.(y[end,:])
        MeanDisplacement = mean(abs, displacements, dims=1)
        StdDisplacement = sqrt.(mean(abs2, displacements, dims=1) .- MeanDisplacement.^2)
        return new(vec(NetDisplacement), vec(MeanDisplacement), vec(StdDisplacement))
    end
end
# TODO: distances in each space

using CSV
df = CSV.read("applications\\electro\\No_EF.csv")

u_obs = reshape(complex.(df[!,:x], df[!,:y]), 37, 50)
t_obs = collect(0.:5.:180.)

export NoEF_Raw, NoEF_Rotated, NoEF_Displacements
NoEF_Raw = NoEF_Experiment(RawTimeSeries(
    u_obs, t_obs),
    t_obs)
NoEF_Rotated = NoEF_Experiment(RotatedTimeSeries(
    u_obs, t_obs),
    t_obs)
NoEF_Displacements = NoEF_Experiment(Displacements(
    u_obs, t_obs),
    t_obs)
