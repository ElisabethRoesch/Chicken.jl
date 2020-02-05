
using JLD, DifferentialEquations, StatsBase, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
include("../src/tools.jl")

datasize = 35
alpha, tspan, solver = 5.0,(0,2.0),Tsit5()
t = range(tspan[1], tspan[2], length = datasize)


train_u0s = [-2.,-1.,0.,1.0,2.0]
ode_data=JLD.load("test/dummy_data/pitchfork.jld")["counts"]

dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t)
ps = n_ode.p
n_epochs = 20
opt1 = Descent(0.005)
function L2_loss_fct()
    counter = 0
    sum = 0
    for i in train_u0s
        counter = counter+1
        sc_level = conv_ts_to_cnt_all_spec(n_ode([i]), Array(range(-3., step = 0.1 , stop = 3)))
        xx = abs.(ode_data[counter] .- sc_level[1])
        #print(xx)
        #print("\n sum\n", sum(xx,1))
        sss = 0
        for i in xx
            sss = i+sss
        end
        sum=sum+sss
    end
    return sum
end
L2_loss_fct()


cb1 = function ()
    println(L2_loss_fct())
end
test_u0s = [-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,-3.]
preds = []
for i in test_u0s
    pred = n_ode([i])
    push!(preds, pred[1,:])
end



print("\nDONE\n")
plot(Array(range(1,stop = datasize)),preds[1])
plot!(Array(range(1,stop = datasize)),preds[2])
plot!(Array(range(1,stop = datasize)),preds[3])
plot!(Array(range(1,stop = datasize)),preds[4])
plot!(Array(range(1,stop = datasize)),preds[5])
plot!(Array(range(1,stop = datasize)),preds[6])
plot!(Array(range(1,stop = datasize)),preds[7])
plot!(Array(range(1,stop = datasize)),preds[8])
plot!(Array(range(1,stop = datasize)),preds[9])
plot!(Array(range(1,stop = datasize)),preds[10])
plot!(Array(range(1,stop = datasize)),preds[11])
plot!(Array(range(1,stop = datasize)),preds[12])
plot!(Array(range(1,stop = datasize)),preds[13])
# train n_ode with collocation method
#@time Flux.train!(L2_loss_fct, ps, data1, opt1, cb = cb1)
DiffEqFlux.sciml_train!(L2_loss_fct, ps, ADAM(0.05), cb = cb1, maxiters = 100)


derivs = []
for i in test_u0s
    d = dudt([i])
    push!(derivs,Flux.data(d)[1])
end
a = test_u0s.+ derivs
plot([1,2],[test_u0s[1], a[1]],label ="", color ="blue", grid =:off)
plot!([1,2],[test_u0s[2], a[2]],label ="", color ="blue")
plot!([1,2],[test_u0s[3], a[3]],label ="", color ="blue")
plot!([1,2],[test_u0s[4], a[4]],label ="", color ="blue")
plot!([1,2],[test_u0s[5], a[5]],label ="", color ="blue")
plot!([1,2],[test_u0s[6], a[6]],label ="", color ="blue")
plot!([1,2],[test_u0s[7], a[7]],label ="", color ="blue")
plot!([1,2],[test_u0s[8], a[8]],label ="", color ="blue")
plot!([1,2],[test_u0s[9], a[9]],label ="", color ="blue")
plot!([1,2],[test_u0s[10], a[10]],label ="", color ="blue")
plot!([1,2],[test_u0s[11], a[11]],label ="", color ="blue")
plot!([1,2],[test_u0s[12], a[12]],label ="", color ="blue")
plot!([1,2],[test_u0s[13], a[13]],label ="", color ="blue")
hline!([-sqrt(alpha),0,sqrt(alpha)], label ="",color ="red")
savefig("alpha_ks_5.pdf")
@save "pitchfork_bifur_alpha_ks_5.bson" dudt
