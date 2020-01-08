using JLD, DifferentialEquations, StatsBase, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
include("../src/tools.jl")
include("../src/example_systems.jl")
datasize = 35
alpha, tspan, solver = 5.0,(0,2.0),Tsit5()
t = range(tspan[1], tspan[2], length = datasize)
train_u0s = [-2.,-1.,0.,1.0,2.0]
ode_data = run_pfsuper_multi_u0(train_u0s)
ran = Array(range(-3., step = 0.1 , stop = 3))
scd = conv_ts_to_cnt_all_spec(ode_data, ran)

for i in 1:5
    xx = reshape(scd[i],length(scd[i]))
    counts_plot = scatter(scd[i,:], ran, color = "green", size = (150,400), grid = "off", label = "Counts")
    plot!(scd[i,:],ran, color = "green", label = "")
    display(counts_plot)
end


JLD.save("test/dummy_data/pitchfork.jld", "counts", scd)
