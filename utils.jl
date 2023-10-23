mutable struct iteration_stats
  primal_residual::Vector{Float64}
  dual_residual::Vector{Float64}
  gap::Float64
  kkt_error::Float64
  relative_primal_residual::Vector{Float64}
  relative_dual_residual::Vector{Float64}
  relative_gap::Float64
  relative_kkt_error::Float64
end

function compute_primal_residual(
  problem::QuadraticProgrammingProblem,
  primal_vec::AbstractVector{Float64},
)
  activities = problem.constraint_matrix * primal_vec

  if isempty(equality_range(problem))
    equality_violation = []
  else
    equality_violation =
      problem.right_hand_side[equality_range(problem)] -
      activities[equality_range(problem)]
  end

  if isempty(inequality_range(problem))
    inequality_violation = []
  else
    inequality_violation =
      max.(
        problem.right_hand_side[inequality_range(problem)] -
        activities[inequality_range(problem)],
        0.0,
      )
  end

  primal_objective = problem.objective_constant + problem.objective_vector' * primal_vec +  0.5 * (primal_vec' * problem.objective_matrix * primal_vec)

  lower_bound_violation = max.(problem.variable_lower_bound - primal_vec, 0.0)
  upper_bound_violation = max.(primal_vec - problem.variable_upper_bound, 0.0)
  return [
    equality_violation
    inequality_violation
    lower_bound_violation
    upper_bound_violation
  ], primal_objective
end

function compute_dual_residual(
    problem::QuadraticProgrammingProblem,
    primal_solution::AbstractVector{Float64},
    dual_solution::AbstractVector{Float64},
  )
    if !isempty(inequality_range(problem))
        dual_residual = max.(-dual_solution[inequality_range(problem)], 0.0)
    else
        dual_residual = []
    end
    primal_gradient = problem.objective_matrix*primal_solution.+problem.objective_vector.-problem.constraint_matrix'*dual_solution
    reduced_costs = compute_reduced_costs_from_primal_gradient(
      problem.variable_lower_bound,
      problem.variable_upper_bound,
      primal_gradient,
    )
    reduced_cost_violations = primal_gradient .- reduced_costs
    dual_residual = [dual_residual; reduced_cost_violations]

    base_dual_objective =
    problem.right_hand_side' * dual_solution + problem.objective_constant -
    0.5 * primal_solution'*problem.objective_matrix* primal_solution

    dual_objective =
    base_dual_objective + reduced_costs_dual_objective_contribution(
      problem.variable_lower_bound,
      problem.variable_upper_bound,
      reduced_costs,
    )

    return dual_residual, dual_objective
end

function compute_reduced_costs_from_primal_gradient(
  variable_lower_bound::Vector{Float64},
  variable_upper_bound::Vector{Float64},
  primal_gradient::AbstractVector{Float64},
)
  primal_size = length(primal_gradient)
  reduced_costs = zeros(primal_size)
  for i in 1:primal_size
    if primal_gradient[i] > 0.0
      bound_value = variable_lower_bound[i]
    else
      bound_value = variable_upper_bound[i]
    end
    if isfinite(bound_value)
      reduced_costs[i] = primal_gradient[i]
    end
  end

  return reduced_costs
end

"""
  reduced_costs_dual_objective_contribution(
    variable_lower_bound,
    variable_upper_bound,
    reduced_costs
  )
This function returns the contribution of the
reduced costs to the dual objective value.
"""
function reduced_costs_dual_objective_contribution(
  variable_lower_bound::Vector{Float64},
  variable_upper_bound::Vector{Float64},
  reduced_costs::Vector{Float64},
)
  dual_objective_contribution = 0.0
  for i in 1:length(variable_lower_bound)
    if reduced_costs[i] == 0.0
      continue
    elseif reduced_costs[i] > 0.0
      # A positive reduced cost is associated with a binding lower bound.
      bound_value = variable_lower_bound[i]
    else
      # A negative reduced cost is associated with a binding upper bound.
      bound_value = variable_upper_bound[i]
    end
    if !isfinite(bound_value)
      return -Inf
    else
      dual_objective_contribution += bound_value * reduced_costs[i]
    end
  end

  return dual_objective_contribution
end


function compute_kkt(
    problem::QuadraticProgrammingProblem,
    primal_solution::AbstractVector{Float64},
    dual_solution::AbstractVector{Float64},
    scaled_problem,
)
  primal_solution::Vector{Float64} =
  primal_solution ./ scaled_problem.variable_rescaling
  dual_solution::Vector{Float64} =
  dual_solution ./ scaled_problem.constraint_rescaling

  # FirstOrderLp.unscale_problem(scaled_problem.scaled_qp,scaled_problem.constraint_rescaling,scaled_problem.variable_rescaling)

  problem = QuadraticProgrammingProblem(
    scaled_problem.original_qp.variable_lower_bound,
    scaled_problem.original_qp.variable_upper_bound,
    scaled_problem.original_qp.objective_matrix,
    scaled_problem.original_qp.objective_vector,
    scaled_problem.original_qp.objective_constant,
    scaled_problem.original_qp.constraint_matrix,
    scaled_problem.original_qp.right_hand_side,
    scaled_problem.original_qp.num_equalities,
  )
  
	primal_residual, primal_obj = compute_primal_residual(problem,primal_solution)
  dual_residual, dual_obj = compute_dual_residual(problem,primal_solution,dual_solution)
  # primal_dual_gap = max( 0.0,
  #     primal_solution'*problem.objective_matrix*primal_solution +
  #     problem.objective_vector'*primal_solution -
  #     problem.right_hand_side'*dual_solution
  # )
  # primal_dual_gap = abs(
  #     primal_solution'*problem.objective_matrix*primal_solution +
  #     problem.objective_vector'*primal_solution -
  #     problem.right_hand_side'*dual_solution
  # )
  gap = abs(primal_obj-dual_obj)
  kkt_error = max(
        maximum(primal_residual),
        maximum(dual_residual),
        gap
    )

  eps_ratio = 1e-3
  eps_gap = 1e-3

  relative_primal_residual = primal_residual ./ (eps_ratio + norm(problem.constraint_matrix*primal_solution, Inf) + norm(problem.right_hand_side,Inf))
  relative_dual_residual = dual_residual ./ (eps_ratio + norm(problem.objective_matrix*primal_solution, Inf) + norm(problem.objective_vector, Inf) + norm(problem.constraint_matrix'*dual_solution, Inf))
  relative_gap = gap / (eps_gap + abs(primal_obj) + abs(dual_obj))

  relative_kkt = max(
    maximum(relative_primal_residual),
    maximum(relative_primal_residual),
    relative_gap
  )

  # primal_residual_scale = primal_residual ./ max(eps_ratio, maximum(abs.(problem.constraint_matrix*primal_solution)),maximum(abs.(problem.right_hand_side)))
  # dual_residual_scale = dual_residual ./ max(eps_ratio, maximum(abs.(problem.objective_matrix*primal_solution)),maximum(abs.(problem.objective_vector)),maximum(abs.(problem.constraint_matrix'*dual_solution)))
  # primal_dual_gap_scale = primal_dual_gap / max(eps_gap, abs(primal_solution'*problem.objective_matrix*primal_solution), abs(problem.objective_vector'*primal_solution), abs(problem.right_hand_side'*dual_solution))

  # kkt_error_scale = max(
  #   maximum(primal_residual_scale),
  #   maximum(dual_residual_scale),
  #   primal_dual_gap_scale
  # )
  
  res = iteration_stats(
      primal_residual,
      dual_residual,
      gap,
      kkt_error,
      relative_primal_residual,
      relative_dual_residual,
      relative_gap,
      relative_kkt,
    )

  # res = iteration_res(
  #   primal_residual,
  #   dual_residual,
  #   primal_dual_gap,
  #   kkt_error,
  #   primal_residual_scale,
  #   dual_residual_scale,
  #   primal_dual_gap_scale,
  #   kkt_error_scale,
  # )
	# return maximum([max.(-b+A*x,0);abs.(Q*x+c+A'*y);max.(-y,0);max.(x'*Q*x+c'*x+b'*y,0)])
    # return sqrt(
    #     sum(primal_residual.^2)+
    #     sum(dual_residual.^2)+
    #     primal_dual_gap^2
    # )
    # return max(
    #     maximum(primal_residual),
    #     maximum(dual_residual),
    #     primal_dual_gap
    # )
    return res
end 


"""
Projects the given point onto a set of bounds.
"""
function projection!(
  primal::AbstractVector{Float64},
  variable_lower_bound::AbstractVector{Float64},
  variable_upper_bound::AbstractVector{Float64},
)
  for idx in 1:length(primal)
    primal[idx] = min(
      variable_upper_bound[idx],
      max(variable_lower_bound[idx], primal[idx]),
    )
  end
end

"""Projects the given primal solution onto the feasible set. That is, all
negative duals for inequality constraints are set to zero."""
function project_primal!(
  primal::AbstractVector{Float64},
  problem::QuadraticProgrammingProblem,
)
  projection!(
    primal,
    problem.variable_lower_bound,
    problem.variable_upper_bound,
  )
end

""" Projects the given dual solution onto the feasible set. """
function project_dual!(
  dual::AbstractVector{Float64},
  problem::QuadraticProgrammingProblem,
)
  for idx in inequality_range(problem)
    dual[idx] = max(dual[idx], 0.0)
  end
end


"""
Estimate the probability that the power method, after k iterations, has relative
error > epsilon.  This is based on Theorem 4.1(a) (on page 13) from
"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf
"""
function power_method_failure_probability(
  dimension::Int64,
  epsilon::Float64,
  k::Int64,
)
  if k < 2 || epsilon <= 0.0
    # The theorem requires epsilon > 0 and k >= 2.
    return 1.0
  end
  return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) *
         sqrt(dimension) *
         (1.0 - epsilon)^(k - 1 / 2)
end

"""
Estimate the maximum singular value using power method
https://en.wikipedia.org/wiki/Power_iteration, returning a result with
desired_relative_error with probability at least 1 - probability_of_failure.
Note that this will take approximately log(n / delta^2)/(2 * epsilon) iterations
as per the discussion at the bottom of page 15 of
"Estimating the Largest Eigenvalue by the Power and Lanczos Algorithms with a
Random Start"
https://pdfs.semanticscholar.org/2b2e/a941e55e5fa2ee9d8f4ff393c14482051143.pdf
For lighter reading on this topic see
https://courses.cs.washington.edu/courses/cse521/16sp/521-lecture-13.pdf
which does not include the failure probability.
# Output
A tuple containing:
- estimate of the maximum singular value
- the number of power iterations required to compute it
"""
function estimate_maximum_singular_value(
  matrix::Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}};
  probability_of_failure = 0.01::Float64,
  desired_relative_error = 0.1::Float64,
  seed::Int64 = 1,
)
  # Epsilon is the relative error on the eigenvalue of matrix' * matrix.
  epsilon = 1.0 - (1.0 - desired_relative_error)^2
  # Use the power method on matrix' * matrix
  x = randn(Random.MersenneTwister(seed), size(matrix, 2))

  number_of_power_iterations = 0
  while power_method_failure_probability(
    size(matrix, 2),
    epsilon,
    number_of_power_iterations,
  ) > probability_of_failure
    x = x / norm(x, 2)
    x = matrix' * (matrix * x)
    number_of_power_iterations += 1
  end

  # The singular value is the square root of the maximum eigenvalue of
  # matrix' * matrix
  return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2)
end


mutable struct AdaptiveStepsize
  step_size::Float64
  step_size_limit::Float64
  #total_number_iterations::Int64
end
"""
Takes a step using the adaptive step size.
It modifies the third argument: solver_state.
"""
function choose_stepsize(
step_params::AdaptiveStepsize,
problem::QuadraticProgrammingProblem,
total_number_iterations::Int64,
current_primal_solution::Vector{Float64},
current_dual_solution::Vector{Float64},
norm_Q, norm_A
)
step_size = step_params.step_size_limit
#total_number_iterations = step_params.total_number_iterations
done = false

reduction_exponent = 0.3
growth_exponent = 0.6

while !done
  next_primal = current_primal_solution - step_size*(problem.objective_matrix*current_primal_solution+problem.objective_vector-problem.constraint_matrix'*current_dual_solution)
  project_primal!(next_primal,problem)

  next_dual = current_dual_solution + step_size*(norm_Q+norm_A)/norm_A*(-problem.constraint_matrix*(2*next_primal-current_primal_solution)+problem.right_hand_side)
  project_dual!(next_dual,problem)

  # movement = (next_primal-current_primal_solution)'*(next_primal-current_primal_solution) - step_size*(next_primal-current_primal_solution)'*problem.objective_matrix*(next_primal-current_primal_solution) + (next_dual-current_dual_solution)'*(next_dual-current_dual_solution)

  movement = (next_primal-current_primal_solution)'*(next_primal-current_primal_solution) + norm_A/(norm_Q+norm_A)* (next_dual-current_dual_solution)'*(next_dual-current_dual_solution)

  interaction = -2*(next_dual-current_dual_solution)'*problem.constraint_matrix*(next_primal-current_primal_solution) + (next_primal-current_primal_solution)'*problem.objective_matrix*(next_primal-current_primal_solution)
  # interaction = abs(interaction)

  # The proof of Theorem 1 requires movement / step_size >= interaction.
  if interaction > 0
    step_size_limit = movement / interaction
  elseif interaction <= 0
    step_size_limit = step_size
  end
  if step_size <= step_size_limit
      step_params.step_size = step_size
      done = true
  end

  first_term =
    (1 - 1/(total_number_iterations + 1)^(reduction_exponent)) * step_size_limit
  second_term =
    (1 + 1/(total_number_iterations + 1)^(growth_exponent)) * step_size
  step_size = min(first_term, second_term)

  if done
      step_params.step_size_limit = step_size
  end

end
#step_params.total_number_iterations += 1
end
