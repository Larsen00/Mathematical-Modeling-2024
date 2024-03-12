using GLPK, Cbc, JuMP, SparseArrays

H = [
10
30
70
50
70
120
140
120
100
80
]


K = [
300 140 40
]


function constructA(H,K)
    h = length(H)
    A = zeros(h,h)
    for i in 1:h
        for j in 1:3
            if (i+j) <= h
                A[i,i+j-1] = K[j]
            end
        end
    end
    for i in 1:h
        for j in 1:i-1
            A[i,j] = A[j,i]  
        end    
    end      
    return A
end

println(constructA(H,K))

# A should be structured as follows
A = [300.0  140.0   40.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
     140.0  300.0  140.0   40.0    0.0    0.0    0.0    0.0    0.0    0.0
      40.0  140.0  300.0  140.0   40.0    0.0    0.0    0.0    0.0    0.0
       0.0   40.0  140.0  300.0  140.0   40.0    0.0    0.0    0.0    0.0
       0.0    0.0   40.0  140.0  300.0  140.0   40.0    0.0    0.0    0.0
       0.0    0.0    0.0   40.0  140.0  300.0  140.0   40.0    0.0    0.0
       0.0    0.0    0.0    0.0   40.0  140.0  300.0  140.0   40.0    0.0
       0.0    0.0    0.0    0.0    0.0   40.0  140.0  300.0  140.0   40.0
       0.0    0.0    0.0    0.0    0.0    0.0   40.0  140.0  300.0  140.0
       0.0    0.0    0.0    0.0    0.0    0.0    0.0   40.0  140.0  300.0
]

[300.0 140.0 40.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
 140.0 300.0 140.0 40.0 0.0 0.0 0.0 0.0 0.0 0.0;
 40.0 140.0 300.0 140.0 40.0 0.0 0.0 0.0 0.0 0.0;
 0.0 40.0 140.0 300.0 140.0 40.0 0.0 0.0 0.0 0.0;
 0.0 0.0 40.0 140.0 300.0 140.0 40.0 0.0 0.0 0.0;
 0.0 0.0 0.0 40.0 140.0 300.0 140.0 40.0 0.0 0.0;
 0.0 0.0 0.0 0.0 40.0 140.0 300.0 140.0 40.0 0.0;
 0.0 0.0 0.0 0.0 0.0 40.0 140.0 300.0 140.0 0.0;
 0.0 0.0 0.0 0.0 0.0 0.0 40.0 140.0 300.0 0.0;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]


function solveIP(H, K)
    h = length(H)
    myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

    A = constructA(H,K)

    @variable(myModel, x[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )

    @objective(myModel, Min, sum(x[j] for j=1:h) )

    @constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
    @constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL
        println("Objective value: ", JuMP.objective_value(myModel))
        println("x = ", JuMP.value.(x))
        println("R = ", JuMP.value.(R))
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end
end

solveIP(H,K)
