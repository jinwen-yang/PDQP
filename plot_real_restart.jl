# adaptive,10,100,1000: QSCTAP2, QSCSD1, QSC205
# apd/pdhg+adaptive/no: QSCTAP2, QSCSD1, QSC205

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
num_iter = 1
num_epoch = Int(total/num_iter)
tol_iter = 1e-6

problem_name = "PRIMALC5"
instance_path = joinpath("/home/jyang20/QP_instance/maros_meszaros", "$(problem_name).SIF")

qp = FirstOrderLp.qps_reader_to_standard_form(instance_path)
original_norm_Q, original_norm_A = estimate_maximum_singular_value(qp.objective_matrix), estimate_maximum_singular_value(qp.constraint_matrix)

m,n = size(qp.constraint_matrix)
scale_problem = FirstOrderLp.rescale_problem(10,true,nothing,0,qp)

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

println("APD with restart:")
freq = [10,100,1000]
color_vec = ["green","orange","red"]

params = RAPDParameters(
    zeros((m+n)), # initial point
    num_iter,     # inner iteration number
    num_epoch,    # outer iteration number
    tol_iter,     # tolerance 
)
kkt_error, x, y = RAPD_adaptive(params,qp,scale_problem)
println("adaptive RAPD: $(kkt_error[end])-$(length(kkt_error))")
println()
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
    xguidefontsize=12,
    yaxis=:log,
    yguidefontsize=12,
    label="APD with adaptive restart"
)


for i in 1:3
    num_iter_2 = freq[i]
    num_epoch_2 = Int(total/num_iter)
    println("restart_freq: $(num_iter_2), total_iter: $(total), tolerance: $(tol_iter)")
    params = RAPDParameters(
        zeros((m+n)), # initial point
        num_iter_2,     # inner iteration number
        num_epoch_2,    # outer iteration number
        tol_iter,     # tolerance 
    )

    kkt_error, x, y = RAPD(params,qp,scale_problem)

    println("APD with restart: $(kkt_error[end])-$(length(kkt_error))")
    println()

    Plots.plot!(
        1:(length(kkt_error)),
        kkt_error,
        linewidth=1,
        color = color_vec[i],
        label="APD with restart length = $(num_iter_2)",
    )
end

Plots.savefig(kkt_error_plt,joinpath("./plot_real", "$(problem_name)_restart_$(tol_iter).png"))