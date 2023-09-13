##load data with DelimitedFiles
function take(dr::DataRunner, n)
    s = dr.start[1]
    s+n-1>length(dr.y) && error("n = $n exceeds the number of points left in the DataRunner ($(length(dr.y)-s+1)).")
    dr.start .+= n
    return dr.X[:, s:s+n-1], dr.y[s:s+n-1]
end

##data loading functions
function initloader(al::ALstruct, dr::DataRunner, ds::DataSet) 
    X, y = take(dr, al.Ninit)
    ds.X = X
    ds.y = y
    return Flux.Data.DataLoader((ds.X, ds.y), batchsize=al.batchsize, shuffle=true)
end

function initvalid(al::ALstruct, drv::DataRunner) 
    X, y = take(drv, al.Nvalid)
    return Flux.Data.DataLoader((X, y), batchsize=al.batchsize, shuffle=false)
end

function getloader(al::ALstruct, dr::DataRunner, ds::DataSet, filterfun) 
    Xsampled, ysampled = take(dr, al.K*al.M)
    ifilter = sortperm(filterfun(Xsampled))[end-al.K+1:end]
    ds.X = hcat(ds.X, Xsampled[:, ifilter])
    ds.y = vcat(ds.y, ysampled[ifilter])
    return Flux.Data.DataLoader((ds.X, ds.y), batchsize=al.batchsize, shuffle=true)
end

function validationloader(al::ALstruct, validpath) 
    X = readdlm("$(validpath)/X_valid.csv", ',')
    y = parse.(Complex{Float64}, readdlm("$(validpath)/y_valid.csv", ',')[:])
    return Flux.Data.DataLoader((X, y), batchsize=al.batchsize, shuffle=false)
end
##