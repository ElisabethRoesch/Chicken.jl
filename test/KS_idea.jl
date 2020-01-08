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


function KS_loss_fct()
    sum = 0.0
    counter = 0
    for i in train_u0s
        counter = counter+1
        s = kolmogorov_smirnov_distance(ode_data[counter[1]], reshape(n_ode([i]),length(n_ode([i]))))
        sum=sum+s
    end
    return sum
end
