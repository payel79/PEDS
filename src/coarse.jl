##compile necessary functions from FDFD and fdfd_local_field
function Maxwell_2d(Lx, Ly, ϵ, ω, dpml, resolution;
    Rpml=1e-20)

    nx = round(Int, Lx * resolution) #nb point in x
    ny = round(Int, (Ly + 2*dpml) * resolution) #nb points in y
    npml = round(Int, dpml*resolution)
    δ = 1/resolution

    # coordinates centered in (0,0)
    x = (1:nx) * δ
    y = (1-npml:ny-npml) * δ

    #define the laplacian operator in x direction
    o = ones(nx)/δ
    # D = spdiagm((-o,o), (-1,0), nx+1, nx)   # v0.6
    Imat, J, V = SparseArrays.spdiagm_internal(-1 => -o, 0 => o);
    D = sparse(Imat, J, V, nx+1, nx)
    ∇2x = transpose(D) * D
    #periodic boundary condition in x direction
    ∇2x[end,1]-=1/δ^2
    ∇2x[1,end]-=1/δ^2

    #define the laplacian operator in y direction
    o = ones(ny) / δ
    σ0 = -log(Rpml) / (4dpml^3/3)
    y′=((-npml:ny-npml) .+ 0.5) * δ

    σ = Float64[ξ>Ly ? σ0 * (ξ-Ly)^2 : ξ<0 ? σ0 * ξ^2 : 0.0 for ξ in y]
    Σ = spdiagm(0 => 1.0 ./(1 .+ (im/ω)*σ))
    σ′ = Float64[ξ>Ly ? σ0 * (ξ-Ly)^2 : ξ<0 ? σ0 * ξ^2 : 0.0 for ξ in y′]
    Σ′ = spdiagm(0 => 1.0 ./(1 .+ (im/ω)*σ′))
    # D = spdiagm((-o, o), (-1, 0), ny+1, ny)   # v0.6
    Imat, J, V = SparseArrays.spdiagm_internal(-1 => -o, 0 => o);
    D = sparse(Imat, J, V, ny+1, ny)
    ∇2y = Σ * transpose(D) * Σ′ * D

    #get 2d laplacian using kronecker product
    Ix = sparse(1.0I, nx, nx)
    Iy = sparse(1.0I, ny, ny)
    ∇2d = (kron(Ix, ∇2y) + kron(∇2x, Iy))

    # geometry = ComplexF64[ϵ(ξ, ζ) for ζ in y, ξ in x]
    geometry = ϵ(x, y)

    return (∇2d - spdiagm(0 => reshape(ω^2 * geometry, length(x)*length(y))),
    nx, ny, x, y)
end

function ϵ_hole_layers(x, y, ps; refractive_indexes=zeros(3), interstice = 0.5, hole = 0.75)
    @assert length(x)> 2 # makes sure δ is defined

    nx, ny = length(x), length(y)
    δ = x[2] - x[1]
    Ly_pml = y[end] - y[1] + δ
    Lx = x[end] - x[1] + δ

    @assert all(ps .> δ) # makes sure pixel-averaging handles all cases
    @assert all(ps .<= Lx) # makes sure that the holes are not bigger than the period

    # material properties of the unit-cell
    if refractive_indexes == zeros(3)
        refractive_index_background = 1.0
        refractive_index_hole = 1.0
        refractive_index_substrate = 1.45
    else
        refractive_index_background, refractive_index_hole,
        refractive_index_substrate = refractive_indexes
    end

    eps_background, eps_hole, eps_substrate =
    refractive_index_background^2, refractive_index_hole^2,
    refractive_index_substrate^2

    geometry = ones(ComplexF64, ny, nx) * eps_background

    index_top_substrate = floor(Int64, Ly_pml * 0.35/ δ) # 80% of the domain is substrate

    # substrate
    geometry[index_top_substrate:end, :] .= eps_substrate

    # handles case for sub-pixel averaging
    if x[nx÷2] == 0
        w_offset=1/2
    else
        w_offset=0
    end

    # holes
    number_holes = length(ps)
    n_inter_hole = floor(Int64, interstice / refractive_index_substrate / δ)
    n_hole_height = floor(Int64, hole / δ)

    @assert index_top_substrate + number_holes * (n_inter_hole + n_hole_height) < ny
    # makes sure that Ly is big enough for the holes (possibly in PML)

    for it_holes = 1:number_holes
        half_width = ps[it_holes]/2δ - w_offset
        n_half_width = floor(Int64, half_width)
        weight_eps_hole = half_width - n_half_width

        # inside holes
        n_start = floor(Int64, (nx - 2*n_half_width)/2 - w_offset)
        geometry[index_top_substrate + it_holes * n_inter_hole + (it_holes-1) *
        n_hole_height + 1:index_top_substrate + it_holes *
        (n_inter_hole + n_hole_height),
        n_start + 1: n_start + floor(Int64, 2*(n_half_width + w_offset)) + 1] .=
        eps_hole

        # pixel averaging
        # left
        geometry[index_top_substrate + it_holes * n_inter_hole + (it_holes-1) *
        n_hole_height + 1:index_top_substrate + it_holes *
        (n_inter_hole + n_hole_height), n_start] .=
        weight_eps_hole * eps_hole + (1 - weight_eps_hole) * eps_substrate
        # right
        geometry[index_top_substrate + it_holes * n_inter_hole + (it_holes-1) *
        n_hole_height + 1:index_top_substrate + it_holes *
        (n_inter_hole + n_hole_height), end-n_start+1] .=
        weight_eps_hole * eps_hole + (1 - weight_eps_hole) * eps_substrate
    end

    return geometry
end

function ϵ_hole_layers_d(x, y, ps, Δ; refractive_indexes=zeros(3), interstice = 0.5, hole = 0.75)
    @assert length(x)> 2 # makes sure δ is defined

    nx, ny = length(x), length(y)
    δ = x[2] - x[1]
    Ly_pml = y[end] - y[1] + δ
    Lx = x[end] - x[1] + δ
    # @assert Lx==cs.Lx

    @assert all(ps .> δ) # makes sure pixel-averaging handles all cases
    @assert all(ps .<= Lx) # makes sure that the holes are not bigger than the period

    # material properties of the unit-cell
    if refractive_indexes == zeros(3)
        refractive_index_background = 1.0
        refractive_index_hole = 1.0
        refractive_index_substrate = 1.45
    else
        refractive_index_background, refractive_index_hole,
        refractive_index_substrate = refractive_indexes
    end

    eps_background, eps_hole, eps_substrate =
    refractive_index_background^2, refractive_index_hole^2,
    refractive_index_substrate^2

    gradient = zeros(ComplexF64, length(ps)) #initializing to 0

    index_top_substrate = floor(Int64, Ly_pml * 0.35/ δ) # 80% of the domain is substrate

    # handles case for sub-pixel averaging
    if x[nx÷2] == 0
        w_offset=1/2
    else
        w_offset=0
    end

    # holes
    number_holes = length(ps)
    n_inter_hole = floor(Int64, interstice / refractive_index_substrate / δ)
    n_hole_height = floor(Int64, hole / δ)

    @assert index_top_substrate + number_holes * (n_inter_hole + n_hole_height) < ny
    # makes sure that Ly is big enough for the holes (possibly in PML)

    @assert isapprox(Ly_pml,cs.Ly + 2*cs.dpml)
    offset = minimum((1:length(y))[cs.dpml.+y.>(0.35 * (cs.Ly + 2*cs.dpml))])-1

    for it_holes = 1:number_holes
        half_width = ps[it_holes]/2δ - w_offset
        coef_deriv = 1/2δ
        n_half_width = floor(Int64, half_width)
        weight_eps_hole = half_width - n_half_width

        # inside holes
        n_start = floor(Int64, (nx - 2*n_half_width)/2 - w_offset)
        # nothing inside holes

        # pixel averaging
        # left
        gradient[it_holes] = sum(Δ[-offset+index_top_substrate + it_holes * n_inter_hole + (it_holes-1) *
        n_hole_height + 1:-offset+index_top_substrate + it_holes *
        (n_inter_hole + n_hole_height), n_start] .*
        coef_deriv * (eps_hole -  eps_substrate))
        # right
        gradient[it_holes]+= sum(Δ[-offset+index_top_substrate + it_holes * n_inter_hole + (it_holes-1) *
        n_hole_height + 1:-offset+index_top_substrate + it_holes *
        (n_inter_hole + n_hole_height), end-n_start+1] .*
        coef_deriv * (eps_hole -  eps_substrate))
    end

    return gradient
end

struct SimulationDomain{T<:Real,P<:Number}
    Lx::T
    Ly::T
    ω::P
    dpml::T
    resolution::Integer
    source::T
    monitor::T
end

function SimulationDomain(Lx::T, Ly::T, ω::P, dpml::T, resolution::Integer, source::T, monitor::T) where {T, P}
    resolution <= 0 && error("The resolution should be a positive integer.")
    (0>source || Ly<source) && error("The source should be inside the computational domain. (0 < source < Ly)")
    (0>monitor || Ly<monitor) && error("The monitor should be inside the computational domain. (0 < monitor < Ly)")
    new(Lx, Ly, ω, dpml, resolution, source, monitor)
end
function SimulationDomain(cs::CSstruct) 
    return SimulationDomain(
        cs.Lx,
        cs.Ly,
        2pi,
        cs.dpml,
        cs.resolution,
        cs.source,
        cs.monitor
    )
end


nx(sd::SimulationDomain) = round(Int, sd.Lx * sd.resolution)
ny(sd::SimulationDomain) = round(Int, (sd.Ly + 2*sd.dpml) * sd.resolution)
npml(sd::SimulationDomain) = round(Int, sd.dpml*sd.resolution)
xs(sd::SimulationDomain) = (1:nx(sd)) * inv(sd.resolution)
ys(sd::SimulationDomain) = (1-npml(sd):ny(sd)-npml(sd)) * inv(sd.resolution)
continuoussource(sd::SimulationDomain) = begin J = zeros(ComplexF64, (ny(sd), nx(sd)))
    @views J[round(Integer, end-(sd.dpml + sd.source) * sd.resolution), :]  .= 1im  * sd.ω * sd.resolution; J end
monitormask(sd::SimulationDomain) = begin M = zeros(Bool, (ny(sd), nx(sd)))
    @views M[round(Integer, end-(sd.dpml + sd.monitor) * sd.resolution), :]  .= true; return M end
maxwelloperator(sd::SimulationDomain, ϵ::AbstractArray) = length(ϵ)!=nx(sd)*ny(sd) ? error("ϵ should have length $(nx(sd)) x $(ny(sd))") : Maxwell_2d(sd.Lx, sd.Ly, (x, y)->ϵ, sd.ω, sd.dpml, sd.resolution)[1]
emfield(sd::SimulationDomain; ϵ::AbstractArray) = begin 
    if debug 
        try 
            return reshape(maxwelloperator(sd, ϵ) \ continuoussource(sd)[:], (ny(sd), nx(sd)))
        catch
            if isleader
                println("saving the matrix of the error")
                writedlm("errortrycatchepsilon", ϵ, ',')
                writedlm("errortrycatch", maxwelloperator(sd, ϵ), ',')
            end
            rethrow()
        end
    end
    return reshape(maxwelloperator(sd, ϵ) \ continuoussource(sd)[:], (ny(sd), nx(sd)))
end
emfield(A, sd::SimulationDomain) = begin
    if debug 
        try
            return reshape(A \ continuoussource(sd)[:], (ny(sd), nx(sd)))
        catch
            if isleader
                println("saving the matrix of the error (2)")
                writedlm("errortrycatch", A, ',')
            end
            rethrow()
        end
    end
    return reshape(A \ continuoussource(sd)[:], (ny(sd), nx(sd)))
end

complextransmission(Ez::AbstractArray, sd::SimulationDomain) = @views mean(Ez[monitormask(sd)])
complextransmission(sd::SimulationDomain, A) = begin Ez=emfield(A, sd); complextransmission(Ez, sd) end
complextransmission(sd::SimulationDomain; ϵ::AbstractArray)= begin Ez=emfield(sd, ϵ=ϵ); complextransmission(Ez, sd) end
complextransmission(ϵ::AbstractArray; sd::SimulationDomain) = complextransmission(sd, ϵ=ϵ) #for chainrule
adjointgradient_complextransmission(sd::SimulationDomain, A, Ez) = sd.ω^2 / nx(sd) * conj.(A' \ monitormask(sd)[:]) .* Ez[:]

struct ResultField{T<:Number,P<:Real,M,N}
    ϵ::Array{Complex{T}, N}
    Ez::Array{Complex{P}, M}
end
function ResultField(ϵ::Array{Complex{T}, N}, Ez::Array{Complex{P}, M}) where {T,P,M,N}
    length(ϵ)!=length(Ez) && error("ϵ and Ez should be of same length")
    new(ϵ, Ez)
end
ϵ!(r::ResultField, ϵ::AbstractArray) = @views @. r.ϵ = ϵ;
Ez!(r::ResultField, Ez::AbstractArray) = @views @. r.Ez = Ez; 

struct SimulationFDFD
    sd::SimulationDomain
    r::ResultField
end
function SimulationFDFD(sd::SimulationDomain)
    SimulationFDFD(sd, ResultField(zeros(ComplexF64, (ny(sd), nx(sd))), zeros(ComplexF64, (ny(sd), nx(sd)))))
end
"""`ϵ!(s::SimulationFDFD, ϵ::AbstractArray)` updates the geometry of the simulationFDFD and recompute the field"""
ϵ!(s::SimulationFDFD, ϵ::AbstractArray) = begin  ϵ!(s.r, ϵ); Ez!(s.r, emfield(s.sd, ϵ=s.r.ϵ)) end

complextransmission(s::SimulationFDFD) = complextransmission(s.r.Ez, s.sd)
complextransmission!(s::SimulationFDFD, ϵ::AbstractArray)= begin s.r.ϵ!=ϵ ? ϵ!(s, ϵ) : nothing; complextransmission(s) end

adjointgradient_complextransmission(s::SimulationFDFD, A) = adjointgradient_complextransmission(s.sd, A, s.r.Ez)

magicindexair(sd, ny_nn)= (sd.dpml.+ys(sd)).<= 0.35 * (sd.Ly + 2*sd.dpml) # 80% of the domain is substrate
magicindexsubstrate(sd, ny_nn) = Bool.(1 .- (magicindexair(sd, ny_nn).+magicindexgeom_nn(sd, ny_nn)))
magicindexgeom_nn(sd, ny_nn)= 0.35 * (sd.Ly + 2*sd.dpml) .< (sd.dpml.+ys(sd)) .<= 0.35 * (sd.Ly + 2*sd.dpml) + ny_nn/sd.resolution 

function buildgeom(sd::SimulationDomain, coarsegeom, substrate_eps::U) where {U}
    ny_nn, _ =size(coarsegeom)
    geom = zeros(ComplexF64, (ny(sd), nx(sd)))
    geom[magicindexair(sd, ny_nn), :] .= one(U)
    geom[magicindexsubstrate(sd, ny_nn), :] .= one(U) * substrate_eps
    geom[magicindexgeom_nn(sd, ny_nn), :] = coarsegeom
    return real.(geom)
end

get_meat(ar, sd) = ar[magicindexgeom_nn(sd, cs.ny_nn), :]

buildgeom(s::SimulationFDFD, coarsegeom, substrate_eps::U) where {U} = buildgeom(s.sd, coarsegeom, substrate_eps)

getfrequency(x) = sum(x[end-2:end] .* Float64[0.5; 0.75; 1.0]) #manual encoding of frequency from AL paper
getfrequencies(x) = map(1:size(x)[end]) do i 
    sum(x[end-2:end, i] .* Float64[0.5; 0.75; 1.0])
end

coarsified_d(p, Δ) = real.(ϵ_hole_layers_d(xs(sd), ys(sd), p, Δ; refractive_indexes=cs.refracsim, interstice = cs.interstice, hole = cs.hole))
coarsified(p) = real.(ϵ_hole_layers(xs(sd), ys(sd), p; refractive_indexes=cs.refracsim, interstice = cs.interstice, hole = cs.hole))

function ChainRules.rrule(::typeof(coarsified), p::AbstractArray) where {T, N}
    function pullback(Δ)
        return (ChainRules.NoTangent(), coarsified_d(p[1:10], Δ))
        #(ChainRules.NoTangent(), coarsified_d(p, get_meat(Δ, sd)))
    end
    
    return coarsified(p), pullback
end

coarse_geom_func(p) = begin
    frequency = getfrequency(p)
    sd_freq = SimulationDomain(cs.Lx, cs.Ly, 2pi*frequency, cs.dpml, cs.resolution, cs.source, cs.monitor)
    geom = coarsified(p[1:10])
    return get_meat(real.(geom), sd_freq), sd_freq
end

function complextransmissionSolver(ϵ::AbstractArray; sd_freq)
    complextransmission(sd_freq, ϵ=buildgeom(sd_freq, ϵ, cs.epssub))/ cs.refsim
end

function ChainRules.rrule(::typeof(complextransmissionSolver), ϵ_val::AbstractArray, f; sd_freq) where {T, N}
    ϵ=buildgeom(sd_freq, ϵ_val, cs.epssub)
    A = maxwelloperator(sd_freq, ϵ)
    if debug && any(isnan, A) && isleader
        writedlm("errorepsilonval", ϵ_val, ',')
        writedlm("errorepsilongeom", ϵ, ',')
    end
    Ez = emfield(A, sd_freq)

    adjointgradient = f.(adjointgradient_complextransmission(sd_freq, A, Ez) ./ cs.refsim)
    adjointgradient = reshape(adjointgradient, (:, cs.nn_x))[magicindexgeom_nn(sd_freq, cs.ny_nn), :]
    
    if debug && any(isnan.(adjointgradient))
        @show (length(adjointgradient), length(findall(isnan, adjointgradient)))
    end

    function pullback(Δ)
        # if debug && isleader
        #     if any(isnan.(adjointgradient .* Δ))
        #         @show (length(Δ), length(findall(isnan, adjointgradient)), length(findall(isnan, Δ)))
        #     end
        # end
        return (ChainRules.NoTangent(), adjointgradient .* Δ)
    end
    
    return f(complextransmission(Ez, sd_freq)/ cs.refsim), pullback
end

function realtransmissionSolver(ϵ::AbstractArray; sd_freq) where {T, N} 
    real(complextransmissionSolver(ϵ, sd_freq=sd_freq))
end
ChainRules.rrule(::typeof(realtransmissionSolver), ϵ::AbstractArray; sd_freq) where {T, N} = ChainRules.rrule(complextransmissionSolver, ϵ, real, sd_freq=sd_freq)
function imagtransmissionSolver(ϵ::AbstractArray; sd_freq) where {T, N} 
    imag(complextransmissionSolver(ϵ, sd_freq=sd_freq))
end
ChainRules.rrule(::typeof(imagtransmissionSolver), ϵ::AbstractArray; sd_freq) where {T, N} = ChainRules.rrule(complextransmissionSolver, ϵ, imag, sd_freq=sd_freq)
