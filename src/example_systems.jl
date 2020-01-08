function run_pfsuper_one_u0(u0)
    x0 = [u0]
    function pfsuper(dx, x, p, t)
        dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan)
    obs = Array(solve(prob, solver,saveat=t))
    return obs[1,:]
end
function run_pfsuper_multi_u0(u0s)
    obs =[]
    for i in u0s
        push!(obs,run_pfsuper_one_u0(i))
    end
    obs
end
