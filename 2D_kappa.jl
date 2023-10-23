using LinearAlgebra
import GZip
import QPSReader
import SparseArrays
import Logging
import Random
import Plots
import FirstOrderLp
using LaTeXStrings
const norm = LinearAlgebra.norm
const opnorm = LinearAlgebra.opnorm
const dot = LinearAlgebra.dot
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
include("solvers_2D.jl")

total = 3000

num_iter = 25
num_epoch = Int(total/num_iter)
tol_iter = 1e-6

# norm_Q, norm_A = 100, 1.0
kappa_Q = 100
norm_Q, norm_A = 100, 1.0
mu_Q = norm_Q / kappa_Q
problem_name = "kappa_$(kappa_Q)"

# qp = FirstOrderLp.QuadraticProgrammingProblem(
#     [0.0],
#     [Inf],
#     [norm_Q;;],
#     [0.0],
#     0,
#     [norm_A;;],
#     [1],
#     1,
# )

qp = FirstOrderLp.QuadraticProgrammingProblem(
    [0.0;0.0],
    [Inf;Inf],
    LinearAlgebra.diagm([norm_Q;mu_Q]),
    [0.0;0.0],
    0,
    norm_A*[1 1;]/sqrt(2),
    [1],
    1,
)

scale_problem = FirstOrderLp.rescale_problem(0,false,nothing,0,qp)

qp = QuadraticProgrammingProblem(
    scale_problem.scaled_qp.variable_lower_bound,
    scale_problem.scaled_qp.variable_upper_bound,
    scale_problem.scaled_qp.objective_matrix,
    scale_problem.scaled_qp.objective_vector,
    scale_problem.scaled_qp.objective_constant,
    scale_problem.scaled_qp.constraint_matrix,
    scale_problem.scaled_qp.right_hand_side,
    scale_problem.scaled_qp.num_equalities,
)

m,n = size(qp.constraint_matrix)
println("$(problem_name):")
println([m;n;qp.num_equalities])
println("restart_freq: $(num_iter), total_iter: $(total), tolerance: $(tol_iter)")
println("norm_Q: $(norm_Q), norm_A: $(norm_A)")

println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
###########################################
################# APD #####################
###########################################
println("APD with restart:")
params = RAPDParameters(
    zeros((m+n)), # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y, last_x_RAPD,last_y_RAPD,ergodic_x_RAPD,ergodic_y_RAPD = RAPD_2D(params,qp,scale_problem)

println("APD with restart: $(kkt_error[end])-$(length(kkt_error))")
println()

# println(kkt_error)

kkt_error_plt = Plots.plot()
Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "blue",
    legend = :topright,
    title = "\$\\mathbf{\\kappa}\$ = $(kappa_Q)",
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="APD with restart"
)




###########################################
########### PDHG with restart #############
###########################################
println("PDHG with restart:")
params = RPDHGParameters(
    zeros((m+n)), # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y, last_x_RPDHG,last_y_RPDHG,ergodic_x_RPDHG,ergodic_y_RPDHG = RPDHG_2D(params,qp,scale_problem)

println("PDHG with restart: $(kkt_error[end])-$(length(kkt_error))")
println()

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


###########################################
########## APD without restart ############
###########################################
println("APD without restart:")
params = RAPDParameters(
    zeros((m+n)), # initial point
    num_iter*num_epoch,     # inner iteration number
    1,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y, last_x_APD,last_y_APD,ergodic_x_APD,ergodic_y_APD = RAPD_2D(params,qp,scale_problem)

println("APD without restart: $(kkt_error[end])-$(length(kkt_error))")
println()


Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "orange",
    legend = :topright,
    # title = "\$\\kappa\$ = $(kappa))",
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="APD without restart"
)



###########################################
############ PDHG no restart ##############
###########################################
println("PDHG without restart:")
params = RPDHGParameters(
    zeros((m+n)), # initial point
    num_iter*num_epoch,     # inner iteration number
    1,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y, last_x_PDHG,last_y_PDHG,ergodic_x_PDHG,ergodic_y_PDHG = RPDHG_2D(params,qp,scale_problem)

println("PDHG without restart: $(kkt_error[end])-$(length(kkt_error))")
println()

# kkt_error_plt = Plots.plot()
Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "red",
    legend = :topright,
    #title = "recipe",
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="PDHG without restart"
)

# Plots.savefig(kkt_error_plt,joinpath("./2D", "$(problem_name)_$(num_iter)_$(num_epoch)_$(params.tol_iter).png"))
Plots.savefig(kkt_error_plt,joinpath("./2D", "$(problem_name)_$(params.tol_iter).png"))

# ########### 2D figure ############
# len = min(length(ergodic_x_RAPD),length(ergodic_x_RPDHG),length(ergodic_x_APD),length(ergodic_x_PDHG))
# iterates_plt = Plots.plot()
# # Plots.plot!(
# #     ergodic_x_RAPD[1:len],
# #     ergodic_y_RAPD[1:len],
# #     markershape = :diamond,
# #     # markersize = 8,
# #     linealpha = 0.0,
# #     markerstrokewidth = 0.0,
# #     color = "blue",
# #     # legend = false,
# #     # xlim=(0.5,1.5),
# #     # ylim=(0.5,1.5),
# #     aspect_ratio=:equal,
# #     xlabel = "x", ylabel = "y",
# #     label="APD with restart"
# # )

# # Plots.plot!(
# #     ergodic_x_RPDHG[1:len],
# #     ergodic_y_RPDHG[1:len],
# #     markershape = :cross,
# #     # markersize = 8,
# #     linealpha = 0.0,
# #     markerstrokewidth = 1.0,
# #     color = "green",
# #     # legend = false,
# #     # xlim=(0.5,1.5),
# #     # ylim=(0.5,1.5),
# #     aspect_ratio=:equal,
# #     xlabel = "x", ylabel = "y",
# #     label="PDHG with restart"
# # )

# Plots.plot!(
#     ergodic_x_APD[1:len],
#     ergodic_y_APD[1:len],
#     markershape = :xcross,
#     # markersize = 8,
#     linealpha = 0.0,
#     markerstrokewidth = 1.0,
#     color = "yellow",
#     # legend = false,
#     # xlim=(0.5,1.5),
#     # ylim=(0.5,1.5),
#     aspect_ratio=:equal,
#     xlabel = "x", ylabel = "y",
#     label="APD without restart"
# )

# Plots.plot!(
#     ergodic_x_PDHG[1:len],
#     ergodic_y_PDHG[1:len],
#     markershape = :circle,
#     # markersize = 8,
#     linealpha = 0.0,
#     markerstrokewidth = 0.0,
#     color = "red",
#     # legend = false,
#     # xlim=(0.5,1.5),
#     # ylim=(0.5,1.5),
#     aspect_ratio=:equal,
#     xlabel = "x", ylabel = "y",
#     label="PDHG without restart"
# )

# Plots.savefig(iterates_plt,joinpath("./2D", "iterates_$(problem_name)_$(num_iter)_$(num_epoch)_$(params.tol_iter).png"))
