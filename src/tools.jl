function conv_ts_to_cnt_one_spec(arr1, ran)
    len = length(ran)
    arr1 = round.(arr1, digits = 1)
    counts = Vector{Float64}(undef, len)
    for i in 1:len
        c = sum(x->x==ran[i], arr1)
        counts[i] = c
    end
    counts
    return counts
end

function conv_ts_to_cnt_all_spec(arrs, ran)
    cnts = []
    for i in 1:size(arrs)[1]
        cnt = conv_ts_to_cnt_one_spec(arrs[i,:], ran)
        push!(cnts, cnt)
    end
    return cnts
end

# reshape(arr1[1],length(arr1[1]))
