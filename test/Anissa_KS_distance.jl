using  StatsBase
# Calculates the KS distance between two distributions (approximate)
function kolmogorov_smirnov_distance(data1::Array{Float64},data2::Array{Float64})
            #Produce function which returns ecdf
            ecdf_func_1 = StatsBase.ecdf(data1)
            ecdf_func_2 = StatsBase.ecdf(data2)
            #find maximum value of both data sets for ecdf intervals
            max = maximum([data1;data2])
            intervals = max/999
            #calculate ecdf value at each interval
            ecdf_vals_1 = Array{Float64,1}(undef,1000)
            for i in 1:1000
                        ecdf_vals_1[i]=ecdf_func_1(intervals*(i-1))
            end
            ecdf_vals_2 = Array{Float64,1}(undef,1000)
            for i in 1:1000
                        ecdf_vals_2[i]=ecdf_func_2(intervals*(i-1))
            end
            dist = maximum(abs.(ecdf_vals_1-ecdf_vals_2))
            return dist
end

# Example call
kolmogorov_smirnov_distance([1.,2.,3.],[1.2,2.,3.])
