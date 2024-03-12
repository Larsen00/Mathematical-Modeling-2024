using GLPK, Cbc, JuMP, SparseArrays
using CSV
using DataFrames
dataframe = DataFrame(CSV.File("channel_data.txt", delim='\t'))
println(dataframe)

