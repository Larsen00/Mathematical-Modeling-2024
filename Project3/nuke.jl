using GLPK, Cbc, JuMP, SparseArrays, DelimitedFiles,HiGHS, Ipopt

# Load the heights
H = readdlm("Project3/interpol_heights.txt")

# Neighbouring removed dirt when bombing at position i
K = [
300 140 40
]

function termination_check(model, problem, x, R, n)
    println("Problem: $problem")
    if termination_status(model) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(model))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        f1 = open("Project3/res/p$(problem)_X.txt", "w")
        f2 = open("Project3/res/p$(problem)_R.txt", "w")
        XX = JuMP.value.(x)
        RR = JuMP.value.(R)
        for i in 1:n
            println(f1, XX[i, :])
            println(f2, RR[i])
        end
        close(f1)
        close(f2)
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end


# Construct the A matrix (Rji)
function constructA(H,K)
    # Make a function that returns A when given H and K
    A = zeros(length(H),length(H))
    l = length(K)
    KK = [reshape(K[end:-1:2], (1,l-1));; K]
    for i in 1:length(H)
        for j in -l+1:l-1
            if i + j >= 1 && i + j <= length(H)
                A[i,i+j] = KK[j+l]
            end
        end
    end
    return A
end

# problem 3 
function solveIP(chd, H, K)
    n = length(H)
    myModel = Model(HiGHS.Optimizer)
    A = constructA(H,K)

    # Variables and expressions
    @variable(myModel, x[1:n], Bin )
    @expression(myModel, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))

    # Objective function
    @objective(myModel, Min, sum(x[j] for j=1:n) )

    # Constraints
    @constraint(myModel, [i=1:n], R[i] >= H[i] + chd )

    # Solve the model
    optimize!(myModel)
    termination_check(myModel, 3, x, R, n)
end


# problem 4
function smooth_channel(chd, K, H)
    m = Model(HiGHS.Optimizer)

    n = length(H)
    A = constructA(H, K)

    # Variables and expressions
    @variable(m, x[1:n], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))
   

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] >= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))

    optimize!(m)
    
    termination_check(m, 4, x, R, n)
end


# Problem 5
function without_neighboring_boms(chd, K, H)
    # Model
    m = Model(HiGHS.Optimizer)

    # length of data points 
    n = length(H)
    A = constructA(H, K)

    # Variables and expressions
    @variable(m, x[1:n], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] >= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))
    @constraint(m, [i=2:n], x[i-1] + x[i] <= 1)

    optimize!(m)
    
    termination_check(m, 5, x, R, n)
end


# Problem 6
function alterK(chd, K1, K2, K3, H)
    # Model
    m = Model(HiGHS.Optimizer)

    # length of data points 
    n = length(H)
    A1 = constructA(H, K1)
    A2 = constructA(H, K2)
    A3 = constructA(H, K3)

    # Variables
    @variable(m, x[1:n, 1:3], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A1[i,j]*x[j,1] + A2[i,j]*x[j,2] + A3[i,j]*x[j,3] for j=1:n))

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] >= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))
    @constraint(m, [i=2:n], sum(x[i-1, j] + x[i, j] for j=1:3) <= 1)
    @constraint(m, [i=1:n], sum(x[i, j] for j= 1:3) <= 1)

    optimize!(m)
    
    termination_check(m, 6, x, R, n)
end



solveIP(10 ,H,K)
smooth_channel(10, K, H)
without_neighboring_boms(10, K, H)

K2 = [500 230 60]
K3 = [1000 400 70]

alterK(10, K, K2, K3, H)



