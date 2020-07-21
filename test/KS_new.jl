function kolmogorov_smirnov_distance(data1,data2)
            ecdf_func_1 = StatsBase.ecdf(data1)
            ecdf_func_2 = StatsBase.ecdf(data2)
            max = maximum([data1;data2])
            intervals = max/999
            ecdf_vals_1 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                        ecdf_vals_1[i]=ecdf_func_1(intervals*(i-1))
            end
            ecdf_vals_2 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                        ecdf_vals_2[i]=ecdf_func_2(intervals*(i-1))
            end
            dist = maximum(abs.(ecdf_vals_1-ecdf_vals_2))
            return dist
end



using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqParamEstim, Plots, StatsBase
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = kolmogorov_smirnov_distance(ode_data, pred)
    #loss = sum(abs2,ode_data .- pred)
    loss,pred
end

xx=[1 2.2 3;1 2 3.]
ecdf(xx[1,:])
xx[1,:]
loss_n_ode(n_ode.p) # n_ode.p stores the initial parameters of the neural ODE


cb = function (p,l,pred) #callback function to observe training
  display(l)
  # pred = predict_n_ode(p)
  # # plot current prediction against data
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...)

res1 = DiffEqFlux.sciml_train!(loss_n_ode, n_ode.p, Descent(0.001), cb = cb, maxiters = 1000)


pl = scatter(t,ode_data[1,:],label="data")
pred =predict_n_ode(n_ode.p)
scatter!(pl,t,pred[1,:],label="prediction")
