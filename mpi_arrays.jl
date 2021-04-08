include(joinpath("MPIArrays.jl", "src", "MPIArrays.jl"))
using .MPIArrays, MPI
using Random

# Set up MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

println("Hello world, I am rank $(rank) of $(size) on $(gethostname())")

MPI.Barrier(comm)

# Size of the matrix
N = 30

# Create an uninitialized matrix and vector
x = MPIArray{Float64}(N)
A = MPIArray{Float64}(N,N)

# Set random values by applying the `rand!` function to each local element in x and A
forlocalpart!(rand!, x)
forlocalpart!(rand!, A)

# Make sure every process finished initializing the coefficients
sync(A, x)
b = A*x

# Print result
println("$(rank): b=$(b)")

# Vector example
y = MPIArray{Float64}(4)

# Each rank get 2 (consecutive) indices into the y-vector
index = rank*2 + 1
yblock = y[index : index + 1]
println("$(rank): $(yblock) / $([i for i in index : index + 1])")


# Get "view" into block
ymat = getblock(yblock)
# Write into view
ymat[1:2] .= rank
# Syncronize changes back to block
putblock!(ymat, yblock)
# Ensure that all ranks have completed the `putblock!` operation
MPI.Barrier(comm)

# Show output
println("$(rank): $(yblock)")

# Show global indexing
gb = GlobalBlock(ymat, yblock)
println("$(rank): gb[$(index)] = $(gb[index])")
