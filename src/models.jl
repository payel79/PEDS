##instantiation of single model from parameters
layerlist(in, out, fun) = [Flux.Dense(u,v,w) for (u,v,w) in zip(in, out, fun)]

function initmodel(nn::NNstruct)
    ##generator model
    mgenlist = []
    push!(mgenlist, layerlist(nn.inGen,nn.outGen,nn.funGen)...)
    push!(mgenlist, nn.postGen...)
    mgen = Flux.Chain(mgenlist...)
    ##combining combining weight
    cw=0.5 # ones(1)/nn.multfact
    ##variance model
    mvarlist = []
    push!(mvarlist, nn.preVar) 
    push!(mvarlist, layerlist(nn.inVar,nn.outVar,nn.funVar)...)
    mvar = Flux.Chain(mvarlist...)
    return (mgen, cw, mvar)
end

function initbase(nn::NNstruct) 
    ##generator model
    mgenlist = []
    push!(mgenlist, layerlist(nn.inGen,nn.outGen,nn.funGen)...)
    mgen = Flux.Chain(mgenlist...)
    ##prediction model (replace solver)
    predlist=[]
    push!(predlist, Flux.Dense(nn.outGen[end],2,tanh))
    push!(predlist, nn.postBase...)
    pred = Flux.Chain(predlist...)
    ##variance model
    mvarlist = []
    push!(mvarlist, Flux.Dense(nn.outGen[end],1,softplus))
    mvar = Flux.Chain(mvarlist...)
    return (mgen, pred, mvar)
end

##ensemble evaluation
ensmean(μs) = mean(μs)

function ensvar(μ1s, μ2s, σs)  
    ensmeansq = ensmean(μ1s)^2 + ensmean(μ2s)^2
    mean(@. σs^2 + (μ1s^2 + μ2s^2 - ensmeansq))
end