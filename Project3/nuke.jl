using GLPK, Cbc, JuMP, SparseArrays, Ipopt, DelimitedFiles

# Load the heights
H = readdlm("Project3/interpol_heights.txt")

# Neighbouring removed dirt when bombing at position i
K = [
300 140 40
]

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

# problem 2 
function solveIP(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )

    @objective(myModel, Min, sum(x[j] for j=1:h) )

    @constraint(myModel, [i=1:h], sum(A[i,j]*x[j] for j=1:h) >= H[i] + 10 )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end


# problem 3
function problem3(chd, K, H)
    m = Model(Cbc.Optimizer)

    n = length(H)
    A = constructA(H, K)

    @variable(m, x[1:n], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] == R[i] - (H[i] + chd))

    optimize!(m)
    
    if termination_status(m) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(m))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimization was not successful. Return code: ", termination_status(m))
    end
end


solveIP(H,K)
problem3(10, K, H)