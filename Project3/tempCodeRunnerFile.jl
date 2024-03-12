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
        for j in 1:i
            A[i,j] = A[j,i]  
        end    
    end      
    return A
end

println(constructA(H,K))