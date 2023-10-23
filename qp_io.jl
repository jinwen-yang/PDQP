"""
A QuadraticProgrammingProblem struct specifies a quadratic programming problem
with the following format:
```
minimize 1/2 x' * objective_matrix * x + objective_vector' * x
          + objective_constant
s.t. constraint_matrix[1:num_equalities, :] * x =
     right_hand_side[1:num_equalities]
     constraint_matrix[(num_equalities + 1):end, :] * x >=
     right_hand_side[(num_equalities + 1):end]
     variable_lower_bound <= x <= variable_upper_bound
```
The variable_lower_bound may contain `-Inf` elements and variable_upper_bound
may contain `Inf` elements when the corresponding variable bound is not present.
"""
mutable struct QuadraticProgrammingProblem
  """
  The vector of variable lower bounds.
  """
  variable_lower_bound::Vector{Float64}

  """
  The vector of variable upper bounds.
  """
  variable_upper_bound::Vector{Float64}

  """
  The symmetric and positive semidefinite matrix that defines the quadratic
  term in the objective.
  """
  objective_matrix::Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}

  """
  The linear coefficients of the objective function.
  """
  objective_vector::Vector{Float64}

  """
  The constant term of the objective function.
  """
  objective_constant::Float64

  """
  The matrix of coefficients in the linear constraints.
  """
  constraint_matrix::Union{SparseMatrixCSC{Float64,Int64},Matrix{Float64}}

  """
  The vector of right-hand side values in the linear constraints.
  """
  right_hand_side::Vector{Float64}

  """
  The number of equalities in the problem. This value splits the rows of the
  constraint_matrix between the equality and inequality parts.
  """
  num_equalities::Int64
end




mutable struct TwoSidedQpProblem
  "Lower bounds on the variables."
  variable_lower_bound::Vector{Float64}
  "Upper bounds on the variables."
  variable_upper_bound::Vector{Float64}
  "Lower bounds on the constraints."
  constraint_lower_bound::Vector{Float64}
  "Upper bounds on the constraints."
  constraint_upper_bound::Vector{Float64}
  "The constraint matrix."
  constraint_matrix::SparseMatrixCSC{Float64,Int64}
  "The constant term in the objective."
  objective_offset::Float64
  "The objective vector."
  objective_vector::Vector{Float64}
  "The objective matrix."
  objective_matrix::SparseMatrixCSC{Float64,Int64}
end

"""
Transforms a quadratic program into a `QuadraticProgrammingProblem` struct.
The `TwoSidedQpProblem` is destructively modified in place to avoid creating a copy.
# Returns
A QuadraticProgrammingProblem struct containing the quadratic programming problem.
"""
function transform_to_standard_form(qp::TwoSidedQpProblem)
  two_sided_rows_to_slacks(qp)

  is_equality_row = qp.constraint_lower_bound .== qp.constraint_upper_bound
  is_geq_row = .!is_equality_row .& isfinite.(qp.constraint_lower_bound)
  is_leq_row = .!is_equality_row .& isfinite.(qp.constraint_upper_bound)

  # Two-sided rows should be removed by two_sided_rows_to_slacks.
  @assert !any(is_geq_row .& is_leq_row)

  num_equalities = sum(is_equality_row)

  if num_equalities + sum(is_geq_row) + sum(is_leq_row) !=
     length(qp.constraint_lower_bound)
    error("Not all constraints have finite bounds on at least one side.")
  end

  # Flip the signs of the leq rows in place.
  # TODO: Skip this pass if there are no leq rows.
  for idx in 1:SparseArrays.nnz(qp.constraint_matrix)
    if is_leq_row[qp.constraint_matrix.rowval[idx]]
      qp.constraint_matrix.nzval[idx] *= -1
    end
  end

  new_row_to_old = [findall(is_equality_row); findall(.!is_equality_row)]
  if new_row_to_old != 1:size(qp.constraint_matrix, 1)
    row_permute_in_place(qp.constraint_matrix, invperm(new_row_to_old))
  end

  right_hand_side = copy(qp.constraint_lower_bound)
  right_hand_side[is_leq_row] .= .-qp.constraint_upper_bound[is_leq_row]
  permute!(right_hand_side, new_row_to_old)

  return QuadraticProgrammingProblem(
    qp.variable_lower_bound,
    qp.variable_upper_bound,
    qp.objective_matrix,
    qp.objective_vector,
    qp.objective_offset,
    qp.constraint_matrix,
    right_hand_side,
    num_equalities,
  )
end

"""
Transforms a `TwoSidedQpProblem` in-place to remove two-sided constraints, i.e.,
constraints with lower and upper bounds that are both finite and not equal.
For each constraint, a slack variable is added, and the constraint is changed
to an equality: `l <= a'x <= u` becomes `a'x - s = 0, l <= s <= u`.
"""
function two_sided_rows_to_slacks(qp::TwoSidedQpProblem)
  two_sided_rows = findall(
    isfinite.(qp.constraint_lower_bound) .&
    isfinite.(qp.constraint_upper_bound) .&
    (qp.constraint_lower_bound .!= qp.constraint_upper_bound),
  )
  if isempty(two_sided_rows)
    return
  end

  slack_matrix = sparse(
    two_sided_rows,  # row indices
    1:length(two_sided_rows),  # column indices
    fill(-1, length(two_sided_rows)),  # nonzeros
    length(qp.constraint_lower_bound),  # number of rows
    length(two_sided_rows),  # number of columns
  )
  qp.variable_lower_bound =
    [qp.variable_lower_bound; qp.constraint_lower_bound[two_sided_rows]]
  qp.variable_upper_bound =
    [qp.variable_upper_bound; qp.constraint_upper_bound[two_sided_rows]]
  qp.objective_vector = [qp.objective_vector; zeros(length(two_sided_rows))]
  qp.constraint_matrix = [qp.constraint_matrix slack_matrix]
  qp.constraint_lower_bound[two_sided_rows] .= 0
  qp.constraint_upper_bound[two_sided_rows] .= 0

  new_num_variables = length(qp.variable_lower_bound)
  row_indices, col_indices, nonzeros = SparseArrays.findnz(qp.objective_matrix)
  qp.objective_matrix = sparse(
    row_indices,
    col_indices,
    nonzeros,
    new_num_variables,
    new_num_variables,
  )
  return
end

"""
Reads an MPS or QPS file using the QPSReader package and transforms it into a
`QuadraticProgrammingProblem` struct.
# Arguments
- `filename::String`: the path of the file. The file extension is ignored,
  except that if the filename ends with ".gz", then it will be uncompressed
  using GZip. Accepted formats are documented in the README of the QPSReader
  package.
- `fixed_format::Bool`: If true, parse as a fixed-format file.
# Returns
A QuadraticProgrammingProblem struct.
"""
function qps_reader_to_standard_form(
  filename::String;
  fixed_format::Bool = false,
)
  if endswith(filename, ".gz")
    io = GZip.gzopen(filename)
  else
    io = open(filename)
  end

  format = fixed_format ? :fixed : :free

  mps = Logging.with_logger(Logging.NullLogger()) do
    QPSReader.readqps(io, mpsformat = format)
  end
  close(io)

  constraint_matrix =
    sparse(mps.arows, mps.acols, mps.avals, mps.ncon, mps.nvar)
  # The reader returns only the lower triangle of the objective matrix. We have
  # to symmetrize it.
  obj_row_index = Int[]
  obj_col_index = Int[]
  obj_value = Float64[]
  for (i, j, v) in zip(mps.qrows, mps.qcols, mps.qvals)
    push!(obj_row_index, i)
    push!(obj_col_index, j)
    push!(obj_value, v)
    if i != j
      push!(obj_row_index, j)
      push!(obj_col_index, i)
      push!(obj_value, v)
    end
  end
  objective_matrix =
    sparse(obj_row_index, obj_col_index, obj_value, mps.nvar, mps.nvar)
  @assert mps.objsense == :notset

  return transform_to_standard_form(
    TwoSidedQpProblem(
      mps.lvar,
      mps.uvar,
      mps.lcon,
      mps.ucon,
      constraint_matrix,
      mps.c0,
      mps.c,
      objective_matrix,
    ),
  )
end

"""
    row_permute_in_place(matrix::SparseMatrixCSC{Float64, Int64},
                         old_row_to_new::Vector{Int64})
Permutes the rows of `matrix` in place (without allocating a new matrix)
according to the map `old_row_to_new`. Assumes and does not verify that
`old_row_to_new` is a permutation.
"""
function row_permute_in_place(
  matrix::SparseMatrixCSC{Float64,Int64},
  old_row_to_new::Vector{Int64},
)
  coefficients = SparseArrays.nonzeros(matrix)
  row_indices = SparseArrays.rowvals(matrix)
  row_coef_tuples = Tuple{Int64,Float64}[]
  for col in 1:size(matrix, 2)
    nonzero_range = nzrange(matrix, col)
    empty!(row_coef_tuples)
    sizehint!(row_coef_tuples, length(nonzero_range))
    for index_in_matrix in nonzero_range
      new_row = old_row_to_new[row_indices[index_in_matrix]]
      push!(row_coef_tuples, (new_row, coefficients[index_in_matrix]))
    end
    # SparseMatrixCSC requires row indices to be sorted within a column. So we
    # do this sort and then replace the terms in the column.
    sort!(row_coef_tuples, by = t -> t[1])
    for (index_in_column, index_in_matrix) in enumerate(nonzero_range)
      new_row, new_coefficient = row_coef_tuples[index_in_column]
      row_indices[index_in_matrix] = new_row
      coefficients[index_in_matrix] = new_coefficient
    end
  end
  return
end

equality_range(problem::QuadraticProgrammingProblem) = 1:problem.num_equalities

function inequality_range(problem::QuadraticProgrammingProblem)
  return (problem.num_equalities+1):size(problem.constraint_matrix, 1)
end