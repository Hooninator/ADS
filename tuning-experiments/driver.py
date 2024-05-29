import argparse
import subprocess
import os
import math


def write_output(args, result_lst):
    
    fname = f"combblas-{args.alg}-{args.code}-N{args.nodes}-{args.matA}x{args.matB}.out"

    file = open(fname, "w")

    for result in result_lst:
        file.write(result[0]+"\n") #cmd
        file.write(result[1]) #output
     
    file.close()
    
    os.rename(f"./{fname}", f"./perlmutter-dat/{fname}")


def get_layers(ppn, nodes):
    n = ppn*nodes
    layers = [1]
    for l in [2, 4, 8, 16, 32, 64, 128, 256]:
        if l>n:
            continue
        grid_size_2d = n // l
        if (round(grid_size_2d**(1/2))**2==grid_size_2d): # perfect square 2d grids
            layers.append(l)
    print("Layers: " + str(layers))
    return layers


def run(args):
    
    combblas_cmd = f" combblas-spgemm {args.alg} $PSCRATCH/matrices/{args.matA}/{args.matA}.mtx $PSCRATCH/matrices/{args.matB}/{args.matB}.mtx {args.code} "
    
    cmd_lst = []

    for exp in range(int(math.log2(args.ppnmin)), int(math.log2(args.ppnmax*2))):
        ppn = 2**exp
        t = 128//ppn
        n = args.nodes * ppn
        if (round(n**(1/2))**2!=n):
            continue
        srun_cmd = f"export OMP_NUM_THREADS={t} && mpirun -n 16"
        if args.alg=="3D":
            layers = get_layers(ppn, args.nodes)
            for l in layers:
                combblas_cmd_tmp = combblas_cmd + str(l) + " " + str(args.permute)
                cmd  = srun_cmd + combblas_cmd_tmp 
                cmd += f" {args.model}"
                cmd_lst.append(cmd)
        else:
            cmd = srun_cmd + combblas_cmd + "1 " + str(args.permute)
            cmd += f" {args.model}"
            cmd_lst.append(cmd)
    
    print(cmd_lst)
    
    result_lst = []
    
    err_log = open(f"err-{args.nodes}-{args.model}.out", 'a')

    for cmd in cmd_lst:
        print(f"Executing {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            result.check_returncode()
        except subprocess.TimeoutExpired:
            print(f"!!!!!{cmd} timed out")
            err_log.write(args.matA+str(args.nodes)+ ":time-out\n")
        except:
            print(result.stderr)
            err_log.write(args.matA+str(args.nodes)+ ":error\n")
    
    err_log.close()

    return result_lst


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str)
    parser.add_argument("--matA", type=str)
    parser.add_argument("--matB", type=str)
    parser.add_argument("--code", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--permute", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--ppnmin", type=int)
    parser.add_argument("--ppnmax", type=int)
    args = parser.parse_args()
    result_lst = run(args)
    #write_output(args, result_lst)
