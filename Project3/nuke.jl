using GLPK, Cbc, JuMP, SparseArrays, DelimitedFiles,HiGHS

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
function solveIP(chd, H, K)
    n = length(H)
    myModel = Model(HiGHS.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:n], Bin )
    @expression(myModel, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))

    @objective(myModel, Min, sum(x[j] for j=1:n) )

    @constraint(myModel, [i=1:n], R[i] >= H[i] + chd )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        f1 = open("Project3/res/p2_X.txt", "w")
        f2 = open("Project3/res/p2_R.txt", "w")
        XX = JuMP.value.(x)
        RR = JuMP.value.(R)
        for i in 1:n
            println(f1, XX[i])
            println(f2, RR[i])
        end
        close(f1)
        close(f2)
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end


# problem 3
function smooth_channel(chd, K, H)
    m = Model(HiGHS.Optimizer)

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
    @constraint(m, [i=1:n], z[i] <= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))

    optimize!(m)
    
    if termination_status(m) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(m))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        f1 = open("Project3/res/p4_X.txt", "w")
        f2 = open("Project3/res/p4_R.txt", "w")
        XX = JuMP.value.(x)
        RR = JuMP.value.(R)
        for i in 1:n
            println(f1, XX[i])
            println(f2, RR[i])
        end
        close(f1)
        close(f2)
    else
        println("Optimization was not successful. Return code: ", termination_status(m))
    end
end




function without_neighboring_boms(chd, K, H)
    # Model
    m = Model(HiGHS.Optimizer)

    # length of data points 
    n = length(H)
    A = constructA(H, K)

    # Variables
    @variable(m, x[1:n], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A[i,j] * x[j] for j=1:n))

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] <= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))
    @constraint(m, [i=2:n], x[i-1] + x[i] <= 1)

    optimize!(m)
    
    if termination_status(m) == MOI.OPTIMAL
        println("without_neighboring_boms")
        println("Objective value: ", JuMP.objective_value(m))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        f1 = open("Project3/res/p5_X.txt", "w")
        f2 = open("Project3/res/p5_R.txt", "w")
        XX = JuMP.value.(x)
        RR = JuMP.value.(R)
        for i in 1:n
            println(f1, XX[i])
            println(f2, RR[i])
        end
        close(f1)
        close(f2)
    else
        println("Optimization was not successful. Return code: ", termination_status(m))
    end
end



function alterK(chd, K1, K2, K3, H)
    # Model
    m = Model(HiGHS.Optimizer)

    # length of data points 
    n = length(H)
    A1 = constructA(H, K1)
    A2 = constructA(H, K2)
    A3 = constructA(H, K3)

    # Variables
    @variable(m, x[1:n], Bin)
    @variable(m, b1[1:n], Bin)
    @variable(m, b2[1:n], Bin)
    @variable(m, b3[1:n], Bin)
    @variable(m, z[1:n] >= 0)  
    @expression(m, R[i=1:n], sum(A1[i,j]*x[j]*b1[j] + A2[i,j]*x[j]*b2[j] + A3[i,j]*x[j]*b3[j] for j=1:n))

    # Adjust the objective to minimize the sum of z, representing the total deviation
    @objective(m, Min, sum(z[i] for i=1:n))

    # Constraint to ensure enough dirt is removed, considering the height and channel depth
    @constraint(m, [i=1:n], R[i] >= H[i] + chd)

    # Constraints to ensure z[i] captures the absolute deviation
    @constraint(m, [i=1:n], z[i] <= R[i] - (H[i] + chd))
    @constraint(m, [i=1:n], z[i] >= -(R[i] - (H[i] + chd)))
    @constraint(m, [i=2:n], x[i-1] + x[i] <= 1)
    @constraint(m, [i=1:n], b1[i] + b2[i] + b3[i] <= 1)

    optimize!(m)
    
    if termination_status(m) == MOI.OPTIMAL
        println("without_neighboring_boms")
        println("Objective value: ", JuMP.objective_value(m))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
        f1 = open("Project3/res/p6_X.txt", "w")
        f2 = open("Project3/res/p6_R.txt", "w")
        XX = JuMP.value.(x)
        RR = JuMP.value.(R)
        for i in 1:n
            println(f1, XX[i])
            println(f2, RR[i])
        end
        close(f1)
        close(f2)
    else
        println("Optimization was not successful. Return code: ", termination_status(m))
    end
end



# solveIP(10 ,H,K)
# smooth_channel(10, K, H)
# without_neighboring_boms(10, K, H)

K2 = [500 230 60]
K3 = [1000 400 70]

alterK(10, K, K2, K3, H)
