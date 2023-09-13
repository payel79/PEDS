using ChainRules
using Zygote
using SparseArrays
using LinearAlgebra
using Random

##Finite difference operator

# compute the first-derivative finite-difference matrix
# for Dirichlet boundaries, given a grid x[:] of x points
# (including the endpoints where the function = 0!).
function sdiff1(x)
    N = length(x) - 2
    dx1 = Float64[1/(x[i+1] - x[i]) for i = 1:N]
    dx2 = Float64[-1/(x[i+1] - x[i]) for i = 2:N+1]
    I, J, V = SparseArrays.spdiagm_internal(0=>dx1, -1=>dx2)
    return sparse(I, J, V, N+1,N)
end
function sdiff1_periodic(x)
    N = length(x) - 2
    dx1 = Float64[1/(x[i+1] - x[i]) for i = 1:N]
    dx2 = Float64[-1/(x[i+1] - x[i]) for i = 2:N+1]
    I, J, V = SparseArrays.spdiagm_internal(0=>dx1, -1=>dx2)
    D = sparse(I, J, V, N+1,N+1)
    D[end, end] = -D[end, end-1]
    D[1, end] = -D[1,1]
    return D
end

function get_position(L, resolution)
    nx = round(Int, L * resolution) #nb points in x
    δ = 1/resolution
    x = (1:nx) * δ
end

# compute the ∇⋅ c ∇ operator for a function c(x,y)
# and arrays x[:] and y[:] of the x and y points,
# including the endpoints where functions are zero
# (i.e. Dirichlet boundary conditions).
function Laplacian(x, y, c; periodicy=true, debug=false, arrayC=false)
    Dx = sdiff1(x)
    Nx = size(Dx,2)
    Dy = periodicy ? sdiff1_periodic(y) : sdiff1(y)
    debug && @show Dy
    Ny = size(Dy,2)
    
    # discrete gradient operator:
    G = [kron(sparse(I,Ny,Ny), Dx); kron(Dy, sparse(I,Nx,Nx))]
    debug && @show size(kron(sparse(I,Ny,Ny), Dx))
    debug && @show size(kron(Dy, sparse(I,Nx,Nx)))
    
    # grids for derivatives in x and y directions
    xp = [0.5*(x[i]+x[i+1]) for i = 1:length(x)-1]
    yp = [0.5*(y[i]+y[i+1]) for i = 1:length(y)-1]
    
    # evaluate c(x)
    if periodicy && arrayC
        debug && @show size(vec([X+Y for X in xp, Y in y[1:end-1]]))
        debug && @show size(vec([X+Y for X in x[2:end-1], Y in yp]) )
        C = spdiagm(0 => c)
    elseif periodicy
        debug && @show size(vec([c(X,Y) for X in xp, Y in y[1:end-1]]))
        debug && @show size(vec([c(X,Y) for X in x[2:end-1], Y in yp]) )
        C = spdiagm(0 => [ vec([c(X,Y) for X in xp, Y in y[1:end-1]]);
                       vec([c(X,Y) for X in x[2:end-1], Y in yp]) ])
    else
        C = spdiagm(0 => [ vec([c(X,Y) for X in xp, Y in y[2:end-1]]);
                       vec([c(X,Y) for X in x[2:end-1], Y in yp]) ])
    end
    
    return -G' * C * G # ∇⋅ c ∇
end

##simulation parameters

struct Simulation
    x
    y
    Ap
    bp
end

function setsimulationparam(Lx, Ly, resolution)
    x = get_position(Lx,resolution)
    y = get_position(Ly,resolution);
    N = length_c(x, y)
    Ap = Dict(i=> Apifun(i, x, y) for i=1:N);
    bp = bpfun(x, y)
    return (x, y, Ap, bp)
end
function setsimulation(Lx, Ly, resolution)
    (x, y, Ap, bp) = setsimulationparam(Lx, Ly, resolution)
    return Simulation(x, y, Ap, bp)
end

## target function

function targetfunc(x, y, c; field=false) 
    dx = x[2]-x[1]
    dy = y[2]-y[1]
    A =  Laplacian(x, y, c, periodicy=true, debug=false, arrayC=true)
    
    n = length(x)
    c1 = c[1:(n-1)*(n-1)]
    c1 = reshape(c1, ((n-1), (n-1)))
    S = zeros(length(x)-2, length(y)-1)
    S[end,:] = -(c1[end, :])/dx^2 #warning might be /dx/dy
    T = Float32.(reshape(A\vec(S), (length(x)-2, length(y)-1)))
    
    iline = length(x[x.<0.5])
    integrand =  c1[iline+1,:] .* (T[iline+1, :]-T[iline, :]) #not sure if it is at iline+1 or iline
#     integrand = @. (T[iline+1, :]-T[iline, :]) #not sure if it is at iline+1 or iline
    fval = sum(integrand)/dx*dy
    
    !field && return fval
    return T, fval
end

targetfunc(c; sim=default_sim) = targetfunc(sim.x, sim.y, c, field=false)

function ChainRules.rrule(::typeof(targetfunc), c; sim=default_sim)
    
    T, fval = targetfunc(sim.x, sim.y, c, field=true)
    
    n = length(sim.x)
    c1 = c[1:(n-1)*(n-1)]
    c1 = reshape(c1, ((n-1), (n-1)))
    
    A = Laplacian(sim.x, sim.y, c, arrayC=true); #TODO reuse the same LU factorization of A as targetfunc
    dx = sim.x[2]-sim.x[1]
    dy = sim.y[2]-sim.y[1]
    gx = zeros((length(sim.x)-2, length(sim.y)-1))
    iline = length(sim.x[sim.x.<0.5])
    gx[iline+1,:] .= c1[iline+1,:]/dx*dy
    gx[iline,:] .= -c1[iline+1,:]/dx*dy;
    λ = A' \ vec(gx)

    fc = spzeros(length(c))
    for j=1:n-1
        fc[(j-1)*(n-1)+iline+1] = (T[iline+1, j]-T[iline, j])/dx*dy #not sure if it is at iline+1 or iline
    end
    
    adjointgradient = (map(i->fc[i] + λ'*(-sim.Ap[i]*vec(T)+ sim.bp[i]), 1:length(c)))
#     adjointgradient = map(i->λ'*(-sim.Ap[i]*vec(T)+ sim.bp[i]), 1:length(c))
    
    function pullback(Δ)
        return (ChainRules.NoTangent(), Float32.(Δ * adjointgradient))
    end
    return fval, pullback
end
Zygote.refresh()

## other helper functions

length_c(x, y) = length_c(length(x), length(y))

function length_c(nx::Integer, ny::Integer)
    return (nx-1)*(ny-1)+(nx-2)*(ny-1)
end

function Apifun(i, x, y)
    N = length_c(x, y)
    perturb=spzeros(N)
    perturb[i]=1
    return Laplacian(x, y, perturb, arrayC=true)
end

function list_non_zero_index(A)
    listnonzeroindex=[]
    for i in eachindex(A)
        if A[i]!=0
            push!(listnonzeroindex, i)
        end
    end
    return listnonzeroindex
end

function bpfun(x, y)
    #source partial derivative
    nx = length(x)
    ny = length(y)
    dx = x[2]-x[1]

    #create match between index of b and index of c
    bc = zeros((nx-2), (ny-1))
    bc[end, :] .= 1
    bc_list = list_non_zero_index(bc)
    
    c1 = zeros( ((nx-1), (ny-1)))
    c1[end, :] .= 1
    c1_list = list_non_zero_index(c1)

    @assert length(bc_list) == length(c1_list)
    matchdict = Dict(c1_list[i] => bc_list[i] for i in eachindex(bc_list))

    #construct bp
    bp = Dict()
    N = length_c(x, y)

    for i=1:length_c(nx, ny)
        perturb=spzeros((nx-2) * (ny-1))
        if i in keys(matchdict)
            perturb[matchdict[i]] = -1/dx^2
        end
        bp[i] = perturb
    end
    return bp
end

## geometry generator
using ImageFiltering

function generate_smoothed_c(x, y)
    
    img = rand(length(x), length(y))
    imgg = imfilter(img, Kernel.gaussian(3));
    imgg = imgg.>0.5
    c1 = (imgg[1:end-1,1:end-1] .+ imgg[2:end,1:end-1]) * 0.5
    c2 = (imgg[2:end-1, 1:end-1] .+ imgg[2:end-1, 2:end]) * 0.5
    c= [vec(c1);vec(c2)].* 0.9 .+ 0.1
end

# import Random.seed! #tis attempt did not work, because get an error, no matching iterate(::MersenneTwister)
# function Random.seed!(seed::Vector{Int64})
#     length(seed)==1 && return Random.seed!(seed[1])
# end

function generate_smoothed_c(seed, resolution=100)
    Random.seed!(seed)
    img = rand(resolution, resolution)
    imgg = imfilter(img, Kernel.gaussian(3));
    imgg = imgg.>0.5
    c1 = (imgg[1:end-1,1:end-1] .+ imgg[2:end,1:end-1]) * 0.5
    c2 = (imgg[2:end-1, 1:end-1] .+ imgg[2:end-1, 2:end]) * 0.5
    c= [vec(c1);vec(c2)].* 0.9 .+ 0.1
end

function generate_smoothed_c_atT(seed, resolution=100)
    Random.seed!(seed)
    img = rand(resolution, resolution)
    imgg = imfilter(img, Kernel.gaussian(3));
    imgg = imgg.>0.5
    return vec(imgg).* 0.9 .+ 0.1
end

function generateX(x; f=generate_smoothed_c)
    N = length(f(1))
    X = zeros(N, length(x))
    for i in eachindex(x)
        X[:,i] = f(x[i])
    end
    return X
end

# function ChainRules.rrule(::typeof(Random.seed!), k)
#     # @show k
#     function pullback(Δ)
#         # @show Δ
#         return (ChainRules.NoTangent(), ChainRules.NoTangent())
#     end
#     return Random.seed!(k)
# end

function create_cOp(N)
    o = ones(N)/2
    Imat, J, V = SparseArrays.spdiagm_internal(1 => o[1:end-1], 0 => o);
    D = sparse(Imat, J, V, N, N)
    Idmat = sparse(1.0I,N,N)
    avgOpx = kron(Idmat[1:end-1, :], D[1:end-1, :])
    avgOpy = kron(D[1:end-1, :], Idmat[2:end-1, :])
    return [avgOpx; avgOpy]
end

# A = rand(5,5)
# c1 = (A[1:end-1,1:end-1] .+ A[2:end,1:end-1]) * 0.5
# c2 = (A[2:end-1, 1:end-1] .+ A[2:end-1, 2:end]) * 0.5
# avgOp = create_cOp(length(A))
# @assert [vec(c1); vec(c2)] == avgOp * vec(A)

function get_c1_c2(c1c2; showflag = false, resolution=100)
    c1 = c1c2[1:(resolution-1)*(resolution-1)]
    c1 = reshape(c1, ((resolution-1), (resolution-1)))
    c2 = c1c2[(resolution-1)*(resolution-1)+1:end]
    c2 = reshape(c2, (resolution-2, resolution-1))

    if showflag
        subplot(121)
        imshow(c1)
        subplot(122)
        imshow(c2)
    end

    return c1, c2
end

function avg_coarse(X; fineresolution=100, coarseresolution=5)
    jumps = fineresolution÷coarseresolution
    X = reshape(X, (fineresolution, fineresolution))
    Xtemp = sum([X[:, i:jumps:end] for i = 1:jumps])/jumps
    Xcoarse = sum([Xtemp[i:jumps:end, :] for i = 1:jumps])/jumps
    return vec(Xcoarse)
end

function avg_coarse(X, fineresolution, coarseresolution) #faster but might mutate
    jumps = fineresolution÷coarseresolution
    X = reshape(X, (fineresolution, fineresolution))
    Xcoarsetemp = zeros((fineresolution, coarseresolution))
    for i = 1:jumps
        Xcoarsetemp .+= @views X[:, i:jumps:end]
    end
    Xcoarsetemp ./= jumps
    Xcoarse = zeros((coarseresolution, coarseresolution))
    for i = 1:jumps
        Xcoarse .+= @views Xcoarsetemp[i:jumps:end, :]
    end
    Xcoarse ./=jumps
            
#     Xtemp = sum([X[:, i:jumps:end] for i = 1:jumps])/jumps
#     Xcoarse = sum([Xtemp[i:jumps:end, :] for i = 1:jumps])/jumps
    return vec(Xcoarse)
end