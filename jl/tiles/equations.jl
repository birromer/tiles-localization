using Random
using ArgParse
using LinearAlgebra
using Plots
#using CSV

function read_poses(filename)
    poses = Vector{Tuple{Float64,Float64,Float64,Float64}}()
    io = open(filename, "r")
    readline(io)  # skip first line with headers

    while !eof(io)
        line = readline(io)
        push!(poses, Tuple(parse.(Float64, split(line, ","))))
    end

    close(io)
    return poses
end

function u(t)
    lx = 10
    ly = 10
    ω = 0.01

    x = lx*cos(ω*t + pi/2)
    y = ly*sin(2(ω*t + pi/2))

    return (x, y, atan(y,x))
end

function gen_param(x::Tuple{Float64,Float64,Float64}, tile_size::Real)
    y_1 = (x[1]/tile_size - floor(x[1]/tile_size))*tile_size;
    y_2 = (x[2]/tile_size - floor(x[2]/tile_size))*tile_size;
    y_3 = ((x[3]-pi/4)/(pi/2) - floor((x[3]-pi/4)/(pi/2))-0.5) * 2*pi/4

    return (y_1, y_2, y_3)
end

function run_poses(dataset_folder, tile_size, ppm)
    poses_filename = "$dataset_folder/gt.csv"
    poses = read_poses(poses_filename)
    num_iters = size(poses)

    for pose in poses
        println(pose)
    end

    img_folder = "$dataset_folder/dataset_tiles/"
end

function run_traj(tile_size, t0, tf, dt)
    errors = Matrix{Float64}(I, 0, 3)

    let t = t0
        while t <= tf
            z = u(t)
            y = gen_param(z, tile_size)

            eq = [
                min(abs(sin(pi*(y[1]-z[1])/tile_size)), abs(sin(pi*(y[1]-z[2])/tile_size))),
                min(abs(sin(pi*(y[2]-z[2])/tile_size)), abs(sin(pi*(y[2]-z[1])/tile_size))),
                min(abs(sin((y[3]-z[3])/tile_size)), abs(cos((y[3]-z[3])/tile_size)))
            ]

            errors = [
                errors;
                eq'
            ]

            t += dt
        end
    end


    return errors

end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--use_traj"
            action = :store_true
            help = "use trajectory function instead of dataset"
        "--t0"
            help = "initial time for trajectory"
            default = 0
        "--tf"
            help = "final time for trajectory"
            default = 10
        "--dt"
            help = "timestep for trajectory"
            default = 0.05
        "--output", "-o"
            arg_type = String
            help = "output file with error over time"
            default = "out.txt"
        "--dataset", "-d"
            arg_type = String
            help = "dataset folder"
            default = "~/ros/data/tiles/report_fully_autonomous/"
        "--tile_size", "-s"
            arg_type = Real
            help = "size of the tiles in meters"
            default = 1.
        "--ppm"
            arg_type = Real
            help = "pixels per meter in the dataset"
            default = 620.48
        "--seed"
            arg_type =Integer
            help = "seed for the random number generator"
            default = 42
    end

    parsed_args = parse_args(s)

    use_traj = parsed_args["use_traj"]
    tile_size::Real = parsed_args["tile_size"]
    Random.seed!(parsed_args["seed"])

    if !use_traj
        dataset_folder = parsed_args["dataset"]
        output_filename = parsed_args["output"]
        ppm::Real = parsed_args["ppm"]

        execution_time  = @elapsed err = run_poses(dataset_folder, tile_size, ppm)
    else
        t0 = parsed_args["t0"]
        tf = parsed_args["tf"]
        dt = parsed_args["dt"]
        execution_time  = @elapsed err = run_traj(tile_size, t0, tf, dt)

        display(plot!(err[:,1], label="eq1"))
        display(plot!(err[:,2], label="eq2"))
        display(plot!(err[:,3], label="eq3"))
        title!("Min value at each dimension")
        xlabel!("time")
        ylabel!("similarity")
        readline()
    end

    println("Finished in $execution_time sec")

    println("\nmax and min values:")
    println("max error x = ", maximum(err[:,1]))
    println("min error x = ", minimum(err[:,1]))

    println("max error y = ", maximum(err[:,2]))
    println("min error y = ", minimum(err[:,2]))

    println("max error θ = ", maximum(err[:,3]))
    println("min error θ = ", minimum(err[:,3]))

    exit(0)
end

main()
