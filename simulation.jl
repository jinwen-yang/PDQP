using LinearAlgebra
import GZip
import QPSReader
import SparseArrays
import Logging
import Random
import Plots
const norm = LinearAlgebra.norm
const dot = LinearAlgebra.dot
const qr = LinearAlgebra.qr
const eigen = LinearAlgebra.eigen
const nzrange = SparseArrays.nzrange
const nnz = SparseArrays.nnz
const nonzeros = SparseArrays.nonzeros
const rowvals = SparseArrays.rowvals
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spdiagm = SparseArrays.spdiagm
const spzeros = SparseArrays.spzeros

include("qp_io.jl")
include("utils.jl")
include("solvers.jl")

function random_normalized(M,N)
    den = (N ^ 0.25)
    ran_seed= 123
    Random.seed!(ran_seed)
    rng = Random.MersenneTwister(ran_seed)
    S = randn(rng, M, N)/den
    return S
end

function random_PD2(N)
    smallVal = 1e-3#1E-3 #1E-6 #1
    S = random_normalized(N,N)
    S = 0.5*(S+S')
    eig_vals, eig_vecs = eigen(S)
    #@show maxEig = maximum(eig_vals)
    minEig = minimum(eig_vals)
    S = S + (abs(minEig) + smallVal)*I(N)
    return S
end

function random_matrix(M,N,R,cond)
    ran_seed= 123
    Random.seed!(ran_seed)
    rng = Random.MersenneTwister(ran_seed)
    tmp = randn(rng, M, M)
    P, RR = qr(tmp)

    ran_seed= 123
    Random.seed!(ran_seed)
    rng = Random.MersenneTwister(ran_seed)
    tmp = randn(rng, N, N)
    Q, RR = qr(tmp)

    ran_seed = 321
    Random.seed!(ran_seed)
    rng = Random.MersenneTwister(ran_seed)
    vec = sort(abs.(randn(rng, R)),rev=true)
    vec = (vec.-vec[end])./(vec[1]-vec[end]) .+ 1/(cond-1)

    D = diagm(vec)
    if M < N
        if R < M 
            D = vcat(D,zeros(M-R,R))
        end
        D = hcat(D,zeros(M,N-R))
    elseif M > N
        if R < N 
            D = hcat(D,zeros(R,N-R))
        end
        D = vcat(D,zeros(M-R,N))
    else
        if R < N 
            D = hcat(D,zeros(R,N-R))
            D = vcat(D,zeros(M-R,N))
        end   
    end

    A = P*D*Q 
    
    return A 
end


##############################
######### simulation #########
##############################
m = 200
n = 100
r = 95
cond = 100

A = random_normalized(m,n)
# A = random_matrix(m,n,r,cond)
xstar, ystar = zeros(n), zeros(m)
B, C = zeros(n,n), zeros(m,m)
scaleBC = 0.1
B = scaleBC*random_PD2(n)
qp = QuadraticProgrammingProblem(
    zeros(n),
    repeat([Inf],n),
    B,
    zeros(n),
    0,
    A,
    zeros(m),
    m
)

problem_name = "simulation"

m,n = size(qp.constraint_matrix)
println("$(problem_name):")
println([m;n;qp.num_equalities])

num_iter = 60000
num_epoch = 10
tol_iter = 1e-6

z0 = randn(m+n)

###########################################
################# APD #####################
###########################################
params = RAPDParameters(
    z0, # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y = RAPD(params,qp)

println(length(kkt_error))


kkt_error_plt = Plots.plot()
Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "blue",
    legend = :topright,
    title = problem_name,
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="APD with restart"
)
# Plots.savefig(kkt_error_plt,joinpath("./figure", "$(problem_name)_$(params.num_iter)_$(params.num_epoch)_$(params.tol_iter).png"))




###########################################
################ PDHG #####################
###########################################
params = RPDHGParameters(
    z0, # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y = RPDHG(params,qp)

println(length(kkt_error))


# kkt_error_plt = Plots.plot()
Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "green",
    legend = :topright,
    #title = "recipe",
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="PDHG with restart"
)
# Plots.savefig(kkt_error_plt,joinpath("./figure", "PDHG_$(problem_name)_$(params.num_iter)_$(params.num_epoch)_$(params.tol_iter).png"))
Plots.savefig(kkt_error_plt,joinpath("./figure", "$(problem_name)_$(params.num_iter)_$(params.num_epoch)_$(params.tol_iter).png"))