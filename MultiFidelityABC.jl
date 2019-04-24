module MultiFidelityABC
include("simmethods.jl")
include("mfabc.jl")
include("produce_figs.jl")
end
include("Viral.jl")
include("Repressilator.jl")
using .MultiFidelityABC