mutable struct RAPDParameters
	"""
	initial point
	"""
	z0::Vector{Float64}
	"""
	number of inner iterations
	"""
	num_iter::Int64
    """
	number of outer iterations
	"""
	num_epoch::Int64
    """
    tolerance
    """
    tol_iter::Float64
end


function APD(params::RAPDParameters,problem::QuadraticProgrammingProblem,scaled_problem)
	z0 = params.z0
	tol_iter = params.tol_iter
    num_iter = params.num_iter

	Q, c, A, b = problem.objective_matrix, problem.objective_vector, -problem.constraint_matrix, -problem.right_hand_side
	m, n = size(A)
	x, y = z0[1:n], z0[(n+1):end]
    norm_Q, norm_A = estimate_maximum_singular_value(Q), estimate_maximum_singular_value(A)
    
    

	relative_kkt_error = []
    # kkt_error_curr = []
	# append!(kkt_error,compute_kkt(problem,x,y))
	
	xx, avg_x, avg_y = x, x, y

    # alpha = 1

	for i in 1:num_iter
        
        beta = (i+1)/2
        # alpha0 = alpha
        # alpha = 0.5*(1+sqrt(1+4*alpha^2))
        theta = (i-1)/i # i/(i+1)
        # tau_x, tau_y = (i+1)/(2*(norm_Q+num_iter*norm_A)), (i+1)/(2*num_iter*norm_A)
        tau_x, tau_y = i/(2*norm_Q+i*norm_A), 1/(norm_A)
        # tau_x, tau_y = 1.98/(2*(norm_Q+norm_A)), 1.98/(2*norm_A)


        #if i > 0.5*params.num_iter
        x_md = (1-1/beta)*avg_x + 1/beta *xx
        # project_primal!(x_md,problem)
        #else
        #    x_md = xx
        #end
        # x_md = (1-1/beta).*avg_x + 1/beta.*xx
        
        # x_md = (1-1/beta).*xx + 1/beta.*z0[1:n]
        # y_md = (1-1/beta).*y + 1/beta.*z0[(n+1):end]
        x = xx
        xx = x - tau_x.*(Q*x_md+c+A'*y)
        project_primal!(xx,problem)

        # x_md = xx + (alpha0-1)/alpha * (xx-avg_x)
        # yy = y
        # y = y + tau_y.*(A*(theta*(xx-x)+xx)-b)
        y = y + tau_y.*(A*(theta*(xx-x)+xx)-b)
        project_dual!(y,problem)

        # y_md = theta*(avg_y-y)+avg_y
        # x_md = theta*(xx-x)+xx
        # x_md = (1-1/beta).*xx + 1/beta.*x
        # y_md = (1-1/beta).*avg_y + 1/beta.*y

        # x = xx
        # xx = x - tau_x.*(Q*x_md+c+A'*y)
        # project_primal!(xx,problem)

		avg_x = (1-1/beta).*avg_x + 1/beta.*xx
		avg_y = (1-1/beta).*avg_y + 1/beta.*y
        
        # append!(kkt_error,compute_kkt(problem,avg_x,avg_y,scaled_problem))
        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)
        # append!(kkt_error_curr,compute_kkt(problem,xx,y,scaled_problem))

        if relative_kkt_error[end] < tol_iter
            break
        end

	end

	return relative_kkt_error,avg_x,avg_y#, kkt_error_curr
end

function RAPD(params::RAPDParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    m, n = size(problem.constraint_matrix)
    relative_kkt_error = []
    # kkt_error_curr = []
    tol_iter = params.tol_iter

    for epoch in 1:params.num_epoch
        relative_kkt_error_inner, avg_x, avg_y = APD(params,problem,scaled_problem)
        relative_kkt_error = [relative_kkt_error;relative_kkt_error_inner]
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

    return relative_kkt_error, params.z0[1:n], params.z0[(n+1):end]#, kkt_error_curr

end


function RAPD_adaptive(params::RAPDParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    z0 = params.z0
	tol_iter = params.tol_iter
    num_iter = params.num_iter
    num_epoch = params.num_epoch
    num_total = num_iter * num_epoch

	Q, c, A, b = problem.objective_matrix, problem.objective_vector, -problem.constraint_matrix, -problem.right_hand_side
	m, n = size(A)
	x, y = z0[1:n], z0[(n+1):end]
    norm_Q, norm_A = estimate_maximum_singular_value(Q), estimate_maximum_singular_value(A)

	relative_kkt_error = []
	
	xx, avg_x, avg_y = x, x, y
    res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
    initial_kkt = res.relative_kkt_error+1
    inner_iter = 1

	for i in 1:num_total
        
        beta = (inner_iter+1)/2
        theta = (inner_iter-1)/inner_iter
        tau_x, tau_y = inner_iter/(2*norm_Q+inner_iter*norm_A), 1/(norm_A)
        # tau_x, tau_y = 1.98/(2*(norm_Q+norm_A)), 1.98/(2*norm_A)

        x_md = (1-1/beta)*avg_x + 1/beta *xx
        x = xx
        xx = x - tau_x.*(Q*x_md+c+A'*y)
        project_primal!(xx,problem)

        y = y + tau_y.*(A*(theta*(xx-x)+xx)-b)
        project_dual!(y,problem)

		avg_x = (1-1/beta).*avg_x + 1/beta.*xx
		avg_y = (1-1/beta).*avg_y + 1/beta.*y
        
        inner_iter += 1

        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)

        if relative_kkt_error[end] < 0.5 * initial_kkt
            xx, y = avg_x, avg_y
            initial_kkt = relative_kkt_error[end]
            inner_iter = 1

        end

        if relative_kkt_error[end] < tol_iter
            break
        end

	end

    res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
    println("primal residual: $(maximum(res.primal_residual))")
    println("dual residual: $(maximum(res.dual_residual))")
    println("optimality gap: $(res.gap)")

    return relative_kkt_error,avg_x,avg_y

end

###########################################
################ PDHG #####################
###########################################
mutable struct RPDHGParameters
	"""
	initial point
	"""
	z0::Vector{Float64}
	"""
	number of inner iterations
	"""
	num_iter::Int64
    """
	number of outer iterations
	"""
	num_epoch::Int64
    """
    tolerance
    """
    tol_iter::Float64
    # """
	# primal step-size
	# """
	# tau_x::Float64
	# """
	# dual step-size
	# """
	# tau_y::Float64
end

function PDHG(params::RPDHGParameters,problem::QuadraticProgrammingProblem,scaled_problem)
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
    # kkt_error_curr = []
	# append!(kkt_error,compute_kkt(problem,x,y))
	
	xx, avg_x, avg_y = x, x, y

	for i in 1:num_iter

        # choose_stepsize(step_params,problem,i,xx,y,norm_Q, norm_A)
        # tau_x, tau_y = step_params.step_size, step_params.step_size*(norm_Q+norm_A)/norm_A
        # println(step_params.step_size)
        
        # y = y + tau_y.*(A*(2*xx-x)-b)
        # project_dual!(y,problem)
        # x = xx
        # xx = x - tau_x.*(Q*x+c+A'*y)
        # project_primal!(xx,problem)

        x = xx
        xx = x - tau_x.*(Q*x+c+A'*y)
        project_primal!(xx,problem)
        y = y + tau_y.*(A*(2*xx-x)-b)
        project_dual!(y,problem)

		avg_x = (1-1/i).*avg_x + 1/i.*xx
		avg_y = (1-1/i).*avg_y + 1/i.*y
        # avg_x = ( sum_tau*avg_x + step_params.step_size*xx )/(sum_tau+step_params.step_size)
        # avg_y = ( sum_tau*avg_y + step_params.step_size*y )/(sum_tau+step_params.step_size)
        # sum_tau += step_params.step_size
        # append!(kkt_error,compute_kkt(problem,avg_x,avg_y,scaled_problem))
        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)
        # append!(kkt_error_curr,compute_kkt(problem,xx,y,scaled_problem))


        if relative_kkt_error[end] < tol_iter
            break
        end

	end

	return relative_kkt_error,avg_x,avg_y#,kkt_error_curr
end

function RPDHG(params::RPDHGParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    m, n = size(problem.constraint_matrix)
    relative_kkt_error = []
    # kkt_error_curr = []
    tol_iter = params.tol_iter

    for epoch in 1:params.num_epoch
        relative_kkt_error_inner, avg_x, avg_y = PDHG(params,problem,scaled_problem)
        relative_kkt_error = [relative_kkt_error;relative_kkt_error_inner]
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
    # println("primal_obj: $(res.)")

    
    return relative_kkt_error, params.z0[1:n], params.z0[(n+1):end]#, kkt_error_curr
end


function RPDHG_adaptive(params::RPDHGParameters,problem::QuadraticProgrammingProblem,scaled_problem)
    z0 = params.z0
	tol_iter = params.tol_iter
    num_iter = params.num_iter
    num_epoch = params.num_epoch
    num_total = num_iter * num_epoch

    Q, c, A, b = problem.objective_matrix, problem.objective_vector, -problem.constraint_matrix, -problem.right_hand_side
    norm_Q, norm_A = estimate_maximum_singular_value(Q), estimate_maximum_singular_value(A)

    tau_x, tau_y = 0.99/(norm_Q+norm_A), 0.99/(norm_A)

	m, n = size(A)
	x, y = z0[1:n], z0[(n+1):end]

	relative_kkt_error = []
    
	xx, avg_x, avg_y = x, x, y
    res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
    initial_kkt = res.relative_kkt_error+1
    inner_iter = 1

	for i in 1:num_total

        x = xx
        xx = x - tau_x.*(Q*x+c+A'*y)
        project_primal!(xx,problem)
        y = y + tau_y.*(A*(2*xx-x)-b)
        project_dual!(y,problem)

		avg_x = (1-1/inner_iter).*avg_x + 1/inner_iter.*xx
		avg_y = (1-1/inner_iter).*avg_y + 1/inner_iter.*y
        
        inner_iter += 1

        res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
        append!(relative_kkt_error,res.relative_kkt_error)

        if relative_kkt_error[end] < 0.5 * initial_kkt# || (i == 0.01*num_total)
            xx, y = avg_x, avg_y
            initial_kkt = relative_kkt_error[end]
            inner_iter = 1
            # println("restart")
        end

        if relative_kkt_error[end] < tol_iter
            break
        end

	end

    res = compute_kkt(problem,avg_x,avg_y,scaled_problem)
    # println([norm(params.z0[1:n]);norm(params.z0[(n+1):end])])
    println("primal residual: $(maximum(res.primal_residual))")
    println("dual residual: $(maximum(res.dual_residual))")
    println("optimality gap: $(res.gap)")
    # println("primal_obj: $(res.)")

    
    return relative_kkt_error, avg_x, avg_y
end


