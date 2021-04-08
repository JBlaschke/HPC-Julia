using MPI: MPI, Comm


function mpi_shared_array(node_comm::Comm, ::Type{T}, sz::Tuple{Vararg{Int}}; owner_rank=0) where T
    node_rank = MPI.Comm_rank(node_comm)
    len_to_alloc = MPI.Comm_rank(node_comm) == owner_rank ? prod(sz) : 0
    win, bufptr = MPI.Win_allocate_shared(T, len_to_alloc, node_comm)

    if node_rank != owner_rank
        len, sizofT, bufvoidptr = MPI.Win_shared_query(win, owner_rank)
        bufptr = convert(Ptr{T}, bufvoidptr)
    end
    win, unsafe_wrap(Array, bufptr, sz)
end


# Set up MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

println("Hello world, I am rank $(rank) of $(size) on $(gethostname())")

MPI.Barrier(comm)

# Create shared array
owner_rank = 1
win, shared_arr =
    mpi_shared_array(comm, Float64, (100, 2); owner_rank=owner_rank)

MPI.Barrier(comm)

# Write data into shared array
if rank == 0
    shared_arr[:, 1] .= 1:100
elseif rank == 1
    shared_arr[:, 2] .= 901:1000
end

MPI.Barrier(comm)

# Check data on a different rank
if rank == 0
    println(shared_arr[:, 2])
end

# Free window into shared array
MPI.free(win)

# Finalize MPI
MPI.Finalize
