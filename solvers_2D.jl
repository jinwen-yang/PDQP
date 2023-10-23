function APD_2D(params::RAPDParameters,problem::QuadraticProgrammingProblem,scaled_problem)
	z0 = params.z0
	tol_iter = params.tol_iter
    num_iter = params.num_iter
    last_x = []
	last_y = []
    ergodic_x = []
	ergodic_y = []
    
	

	Q, c, A, b = problem.objective_matrix, problem.objective_vector, -problem.constraint_matrix, -problem.right_hand_side
	m, n = size(A)
	x, y = z0[1:n], z0[(n+1):end]
    norm_Q, norm_A = estimate_maximum_singular_value(Q), estimate_maximum_singular_value(A)

	relative_kkt_error = []
    
	xx, avg_x, avg_y = x, x, y
    append!(last_x,xx[1])
    append!(last_y,y[1])
    append!(ergodic_x,avg_x[1])
    append!(ergodic_y,avg_y[1])

	for i in 1:num_iter
        
        beta = (i+1)/2
        theta = (i-1)/i 
        tau_x, tau_y = i/(2*norm_Q+i*norm_A), 1/(norm_A)
        # tau_x, tau_y = 1.98/(2*(norm_Q+norm_A)), 1.98/(2*norm_A)

        x_md = (1-1/beta)*avg_x + 1/beta *xx
        x = xx
        xx = x - tau_x.*(Q*x_md+c+A'*y)
        project_primal!(xx,problem)

        y = y + tau_y.*(A*(theta*(xx-x)+xx)-b)
        project_dual!(y,problem)

		avg_x = (1-1/beta).*avg_x + 1/beta.*xx
		avg_y = (1-1/beta).*avg_y + 1/beta.*y
        
        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)
        append!(last_x,xx[1])
        append!(last_y,y[1])
        append!(ergodic_x,avg_x[1])
        append!(ergodic_y,avg_y[1])

        if relative_kkt_error[end] < tol_iter
            break
        end

	end

	return relative_kkt_error,avg_x,avg_y,last_x,last_y,ergodic_x,ergodic_y
end

function RAPD_2D(params::RAPDParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    m, n = size(problem.constraint_matrix)
    relative_kkt_error = []
    last_x = []
    last_y = []
    ergodic_x = []
    ergodic_y = []
    # kkt_error_curr = []
    tol_iter = params.tol_iter

    for epoch in 1:params.num_epoch
        relative_kkt_error_inner, avg_x, avg_y,last_x_inner,last_y_inner,ergodic_x_inner,ergodic_y_inner = APD_2D(params,problem,scaled_problem)
        relative_kkt_error = [relative_kkt_error;relative_kkt_error_inner]
        last_x = [last_x;last_x_inner]
        last_y = [last_y;last_y_inner]
        ergodic_x = [ergodic_x;ergodic_x_inner]
        ergodic_y = [ergodic_x;ergodic_y_inner]

        
        # kkt_error_curr = [kkt_error_curr;kkt_error_curr_inner]
        params.z0 = [avg_x;avg_y]

        if relative_kkt_error[end] < tol_iter
            break
        end
    end
    res = compute_kkt(problem,params.z0[1:n],params.z0[(n+1):end],scaled_problem)
    # println([norm(params.z0[1:n]);norm(params.z0[(n+1):end])])
    println("primal residual: $(maximum(res.primal_residual))")
    println("dual residual: $(maximum(res.dual_residual))")
    println("optimality gap: $(res.gap)")

    return relative_kkt_error, params.z0[1:n], params.z0[(n+1):end],last_x,last_y,ergodic_x,ergodic_y
end

###########################################
################ PDHG #####################
###########################################
function PDHG_2D(params::RPDHGParameters,problem::QuadraticProgrammingProblem,scaled_problem)
	z0 = params.z0
	tol_iter = params.tol_iter
    num_iter = params.num_iter

    # initial_step_size = 1/maximum(problem.constraint_matrix)
    # step_params = AdaptiveStepsize(initial_step_size,initial_step_size)
    # sum_tau = 0.0

    Q, c, A, b = problem.objective_matrix, problem.objective_vector, -problem.constraint_matrix, -problem.right_hand_side
    norm_Q, norm_A = estimate_maximum_singular_value(Q), estimate_maximum_singular_value(A)


    tau_x, tau_y = 0.99/(norm_Q+norm_A), 0.99/(norm_A)

	m, n = size(A)
	x, y = z0[1:n], z0[(n+1):end]

	relative_kkt_error = []
    last_x = []
	last_y = []
    ergodic_x = []
	ergodic_y = []
    # kkt_error_curr = []
	# append!(kkt_error,compute_kkt(problem,x,y))
	
	xx, avg_x, avg_y = x, x, y
    append!(last_x,xx[1])
    append!(last_y,y[1])
    append!(ergodic_x,avg_x[1])
    append!(ergodic_y,avg_y[1])

	for i in 1:num_iter

        x = xx
        xx = x - tau_x.*(Q*x+c+A'*y)
        project_primal!(xx,problem)
        y = y + tau_y.*(A*(2*xx-x)-b)
        project_dual!(y,problem)

		avg_x = (1-1/i).*avg_x + 1/i.*xx
		avg_y = (1-1/i).*avg_y + 1/i.*y
        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)
        append!(last_x,xx[1])
        append!(last_y,y[1])
        append!(ergodic_x,avg_x[1])
        append!(ergodic_y,avg_y[1])
    
        if relative_kkt_error[end] < tol_iter
            break
        end

	end

	return relative_kkt_error,avg_x,avg_y,last_x,last_y,ergodic_x,ergodic_y
end

function RPDHG_2D(params::RPDHGParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    m, n = size(problem.constraint_matrix)
    relative_kkt_error = []
    last_x = []
    last_y = []
    ergodic_x = []
    ergodic_y = []
    tol_iter = params.tol_iter

    for epoch in 1:params.num_epoch
        relative_kkt_error_inner, avg_x, avg_y, last_x_inner,last_y_inner,ergodic_x_inner,ergodic_y_inner = PDHG_2D(params,problem,scaled_problem)
        relative_kkt_error = [relative_kkt_error;relative_kkt_error_inner]
        last_x = [last_x;last_x_inner]
        last_y = [last_y;last_y_inner]
        ergodic_x = [ergodic_x;ergodic_x_inner]
        ergodic_y = [ergodic_x;ergodic_y_inner]
        
        params.z0 = [avg_x;avg_y]

        if relative_kkt_error[end] < tol_iter
            break
        end
    end
    res = compute_kkt(problem,params.z0[1:n],params.z0[(n+1):end],scaled_problem)
    println("primal residual: $(maximum(res.primal_residual))")
    println("dual residual: $(maximum(res.dual_residual))")
    println("optimality gap: $(res.gap)")

    
    return relative_kkt_error, params.z0[1:n], params.z0[(n+1):end],last_x,last_y,ergodic_x,ergodic_y
end