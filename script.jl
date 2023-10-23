using LinearAlgebra
import GZip
import QPSReader
import SparseArrays
import Logging
import Random
import Plots
import FirstOrderLp
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

total = 200000

num_iter = 1000
num_epoch = Int(total/num_iter)
tol_iter = 1e-6

problem_name =  "QBORE3D"#"CVXQP1_M" #  "BOYD1" "BOYD2" "CONT-200" "CONT-201" "CONT-300"
instance_path = joinpath("/home/jyang20/QP_instance/maros_meszaros", "$(problem_name).SIF")

# qp = qps_reader_to_standard_form(instance_path)
qp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
original_norm_Q, original_norm_A = estimate_maximum_singular_value(qp.objective_matrix), estimate_maximum_singular_value(qp.constraint_matrix)
# tmp = 1.0*I(n)
# qp.constraint_matrix = vcat(qp.constraint_matrix,tmp[findall(qp.variable_lower_bound.>-Inf),:])
# qp.right_hand_side = [qp.right_hand_side;qp.variable_lower_bound]
# qp.variable_lower_bound .= -Inf




# qp.constraint_matrix = [qp.constraint_matrix[1:qp.num_equalities,:]; -qp.constraint_matrix[1:qp.num_equalities,:]; qp.constraint_matrix[(qp.num_equalities+1):end,:]]
# qp.right_hand_side = [qp.right_hand_side[1:qp.num_equalities]; -qp.right_hand_side[1:qp.num_equalities];qp.right_hand_side[(qp.num_equalities+1):end]]
# qp.num_equalities = 0

m,n = size(qp.constraint_matrix)
# qp.right_hand_side = qp.right_hand_side[inequality_range(qp)]
# qp.constraint_matrix = qp.constraint_matrix[inequality_range(qp),:]
# qp.num_equalities = 0 # size(qp.constraint_matrix,1)

# qp.constraint_matrix = SparseMatrixCSC(zeros(m,n))
# qp.right_hand_side = zeros(m)
# qp.num_equalities = m

scale_problem = FirstOrderLp.rescale_problem(10,true,nothing,0,qp)
# qp = scale_problem.scaled_qp



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


println("$(problem_name):")
println([m;n;qp.num_equalities])
println("restart_freq: $(num_iter), total_iter: $(total), tolerance: $(tol_iter)")
println("original_norm_Q: $(original_norm_Q), original_norm_A: $(original_norm_A)")
norm_Q, norm_A = estimate_maximum_singular_value(qp.objective_matrix), estimate_maximum_singular_value(qp.constraint_matrix)
println("norm_Q: $(norm_Q), norm_A: $(norm_A)")
rank_Q, rank_A = LinearAlgebra.rank(qp.objective_matrix), LinearAlgebra.rank(qp.constraint_matrix)
println("rank_Q: $(rank_Q), rank_A: $(rank_A)")
nnz_Q, nnz_A = nnz(qp.objective_matrix), nnz(qp.constraint_matrix)
println("nnz_Q: $(nnz_Q), nnz_A: $(nnz_A)")
println("nnz_density_Q: $(nnz_Q/n/n), nnz_density_A: $(nnz_A/m/n)")

println("unconstrained variables: $(length( findall( (qp.variable_lower_bound.==-Inf) .& (qp.variable_upper_bound.==Inf))))")
println("lower bounded variables: $(length( findall( (qp.variable_lower_bound.>-Inf) .& (qp.variable_upper_bound.==Inf))))")
println("upper bounded variables: $(length( findall( (qp.variable_lower_bound.==-Inf) .& (qp.variable_upper_bound.<Inf))))")
println("two-sided bounded variables: $(length( findall( (qp.variable_lower_bound.>-Inf) .& (qp.variable_upper_bound.<Inf))))")

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

kkt_error, x, y = RAPD(params,qp,scale_problem)

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
########### PDHG with restart #############
###########################################
println("PDHG with restart:")
params = RPDHGParameters(
    zeros((m+n)), # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)

kkt_error, x, y = RPDHG(params,qp,scale_problem)

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
# Plots.savefig(kkt_error_plt,joinpath("./figure", "PDHG_$(problem_name)_$(params.num_iter)_$(params.num_epoch)_$(params.tol_iter).png"))

Plots.savefig(kkt_error_plt,joinpath("./figure", "$(problem_name)_$(num_iter)_$(num_epoch)_$(params.tol_iter)_restart.png"))


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

kkt_error, x, y = RAPD(params,qp,scale_problem)

println("APD without restart: $(kkt_error[end])-$(length(kkt_error))")
println()


Plots.plot!(
    1:(length(kkt_error)),
    kkt_error,
    linewidth=1,
    color = "yellow",
    legend = :topright,
    title = problem_name,
    xlabel = "iterations",
    ylabel = "KKT error",
    #xaxis=:log,
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="APD without restart"
)

Plots.savefig(kkt_error_plt,joinpath("./figure", "$(problem_name)_$(num_iter)_$(num_epoch)_$(params.tol_iter)_restart_momentum.png"))


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

kkt_error, x, y = RPDHG(params,qp,scale_problem)

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

Plots.savefig(kkt_error_plt,joinpath("./figure", "$(problem_name)_$(num_iter)_$(num_epoch)_$(params.tol_iter).png"))