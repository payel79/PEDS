##define the neural network parameters
Base.@kwdef struct NNstruct
    ##parameters of generator nn
    inGen = [13, 256, 256]
    outGen = [256, 256, 19*221]
    funGen = [relu, relu, hardtanh]
    postGen = [x-> @. x*1.5 + 2.5; x-> reshape(x, (221,19,:))]
    ##combining parameter
    multfact = 100
    ##parameters of variance nn
    preVar = x-> reshape(x, (:,))
    inVar = [4199, 256, 256, 256]
    outVar = [256, 256, 256, 1]
    funVar = [relu, relu, relu, relu]#x-> (x<20 ? softplus(x) : x)]
    ##baseline
    postBase = [x-> @. 1.3 * x]
end
##define the coarse solver parameters
Base.@kwdef struct CSstruct
    resolution = 20
    Lx = 0.95
    Ly = 17.0
    dpml = 2.0
    source = 1.0
    monitor = 16.0
    epssub = 1.45^2
    refracsim = Float64[1.0, 1.0, 1.45]
    ny_nn = 221 
    nn_x = 19
    interstice = 0.5
    hole = 0.75
    refsim = -0.33133778612182957 + 0.12500380630233138im #for resolution=20
end
##define active learning type
Base.@kwdef struct ALstruct
    J::Int=5
    T::Int=8
    Ninit::Int=256
    K::Int=64
    M::Int=4
    ne::Int=10
    batchsize::Int=64
    η = 1e-3
    Nvalid::Int=512
end

struct DataRunner
    X 
    y
    start
end

Base.@kwdef mutable struct DataSet
    X = []
    y = []
end
##
