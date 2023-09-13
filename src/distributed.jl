##distributed functionalities
function sum_reduce(comm, localsum)
    MPI.Allreduce(localsum, MPI.SUM, comm)
end

function ChainRules.rrule(::typeof(sum_reduce), comm, localsum)
    function pullback(Δ)
        return (ChainRules.NoTangent(), ChainRules.NoTangent(), Δ) 
    end  
    return sum_reduce(comm, localsum), pullback
end

function ChainRules.rrule(::typeof(MPI.Comm_size), comm)
    function pullback(Δ)
        return (ChainRules.NoTangent(), ChainRules.NoTangent())
    end  
    return MPI.Comm_size(comm), pullback
end

function ChainRules.rrule(::typeof(MPI.Comm_rank), comm)
    function pullback(Δ)
        return (ChainRules.NoTangent(), ChainRules.NoTangent())
    end  
    return MPI.Comm_rank(comm), pullback
end

# function dMSE(comm, mloglik, m, x, y) """dMSE computes the MSE using parallelization over the batch with MPI. Assumes a model called m""" 
#     s = 0.
#     for i=1+MPI.Comm_rank(comm):MPI.Comm_size(comm):length(y)
#         rp, ip = m(x[:,i])
#         s = s + (rp-real(y[i]))^2 + (ip-imag(y[iå]))^2
#     end
#     return sum_reduce(comm, s) / length(y)
# end
#chnged for Fourier
function dMSE(comm, mloglik, m, x, y) """dMSE computes the MSE using parallelization over the batch with MPI. Assumes a model called m""" 
    # s = 0.
    # for i=1+MPI.Comm_rank(comm):MPI.Comm_size(comm):length(y)
    #     rp, ip = m(x[:,i])
    #     s = s + (rp-real(y[i]))^2 + (ip-imag(y[i]))^2
    # end
    nvals = length(y)÷MPI.Comm_size(comm)
    s = nvals * Flux.Losses.mse(m(x[:, MPI.Comm_rank(comm)*nvals+1:(MPI.Comm_rank(comm)+1)*nvals]), 
    y[MPI.Comm_rank(comm)*nvals+1:(MPI.Comm_rank(comm)+1)*nvals])
    mseval = (sum_reduce(comm, s) / length(y))
    return mseval
end

function dHuber(comm, mloglik, m, x, y) """dMSE computes the MSE using parallelization over the batch with MPI. Assumes a model called m""" 
    # s = 0.
    # for i=1+MPI.Comm_rank(comm):MPI.Comm_size(comm):length(y)
    #     rp, ip = m(x[:,i])
    #     s = s + (rp-real(y[i]))^2 + (ip-imag(y[i]))^2
    # end
    nvals = length(y)÷MPI.Comm_size(comm)
    s = nvals * Flux.Losses.huber_loss(m(x[:, MPI.Comm_rank(comm)*nvals+1:(MPI.Comm_rank(comm)+1)*nvals]), 
    y[MPI.Comm_rank(comm)*nvals+1:(MPI.Comm_rank(comm)+1)*nvals], δ=1e-3)
    mseval = (sum_reduce(comm, s) / length(y))
    return mseval
end


function dNLL(comm, mloglik, m, x, y) """dMSE computes the NLL (negative log likelihood) using parallelization over the batch with MPI. Assume a model called mloglik""" 
    s = 0.
    for i=1+MPI.Comm_rank(comm):MPI.Comm_size(comm):length(y)
        rp, ip, vp = mloglik(x[:,i])
        vp += 1e-6 # to avoid division by zero
        sadd = log(vp) + ((rp-real(y[i]))^2 + (ip-imag(y[i]))^2)/2/vp^2
        s += sadd
    end
    nllval = sum_reduce(comm, s) / length(y)
    debug && isleader &&  @show (model_color, nllval)
    return nllval
end
Zygote.refresh()


function dFE(comm, commModel, commLeader, m) """dFE computes the FE using parallelization over the batch with MPI""" 
    evalsr = zeros(al.Nvalid)
    evalsi = zeros(al.Nvalid)
    FE = 0.
    j=0
    ys = Complex{Float64}[]
    for (x, y) in valid
        for i=1+MPI.Comm_rank(commModel):MPI.Comm_size(commModel):length(y)
            rp, ip = m(x[:,i])
            evalsr[j*length(y)+i] = rp
            evalsi[j*length(y)+i] = ip
        end
        j+=1
        push!(ys, y...)
    end
    evalsrModel = sum_reduce(commModel, evalsr)
    evalsiModel = sum_reduce(commModel, evalsi)
    evalsr = sum_reduce(commLeader, evalsrModel) / al.J
    evalsi = sum_reduce(commLeader, evalsiModel) / al.J
    if MPI.Comm_rank(comm) == 0
        ŷ = @. evalsr + 1im * evalsi
        FE = norm(ŷ - ys)/norm(ys)
        @show FE
    end
    return FE
end


function varfilter(mloglik, X) "varfilter collect the evaluations of models and computes the pulled variance."
    outputsreduced = zeros((3*min(al.J, MPI.Comm_size(comm)), size(X)[end]))
    outputs = zeros((3*min(al.J, MPI.Comm_size(comm)), size(X)[end]))
    for i=1+MPI.Comm_rank(commModel):MPI.Comm_size(commModel):size(X)[end]
        if isleader
            outputs[1+3*MPI.Comm_rank(commLeader):3+3*MPI.Comm_rank(commLeader), i] = mloglik(X[:,i])
        end
    end
    outputsreduced = sum_reduce(commModel, outputs)
    outputs = sum_reduce(commLeader, outputsreduced)
    uncertainties = zeros(size(X)[end])
    if MPI.Comm_rank(comm) == 0
        uncertainties = map(i->ensvar(outputs[1:3:end, i], outputs[2:3:end, i], outputs[3:3:end, i]), 1:size(X)[end])
    end
    MPI.Bcast!(uncertainties, 0, comm)
    return uncertainties
end