# module PEDS

##import julia packages
using Random
using Flux#: Dense, Chain
using NNlib#: relu, softplus
using StatsBase
using DelimitedFiles#: readdlm, writedlm
using MPI
using ChainRules
using Zygote
using BSON#: @save
using LinearAlgebra
using SparseArrays
using ChangePrecision
using CUDA
##Initialize MPI
MPI.Init()
##include functionalities broken up in separate files


include("objects.jl")
include("coarse.jl")
include("data.jl")
include("models.jl")
include("distributed.jl")
##
# end
