using Distributed
using DelimitedFiles
@everywhere include("mfabc.jl")

function get_benchmark(N::Int64,outfile::String="./output.txt")
    output = pmap(i->runpair(i,nc=3.0,epsilon=0.01), 1:N)
    writedlm(outfile,output)
end