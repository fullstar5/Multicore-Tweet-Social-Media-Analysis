# Multicore-Tweet-Social-Media-Analysis

**What we did**
<p>we implemented a program that processes 1MB, 50MB and 100GB of
data through implementing a parallelized application, then intends to find out the most
active/happiest hour and day ever. During the implementation, we found out that brutally
iterating through a 100GB file will be extremely time consuming. Hence, with the
combination of “Amdahl's Law” and “Gustafson-Barsis’ Law”, various optimizations are
being developed and shown in this report. As 1MB file is too small to make comparison
between approaches, we are only going to demonstrate performance on 50MB and 100GB
files</p>

**How to Invoke Program**
<p>Three different slurm scripts that run a certain approach with node resources are attached
with code. The script identifies the version of Python and mpi4py, resources will be used, and
includes which approach will be running. To run the script on Spartan, first need to change
the approach python file identified within the script, then using ‘sbatch’ command followed
by script file name, for example, ‘sbatch 1node1task.slurm’. To run scripts for different sizes
of file (100GB, 50MB), you need to change the file path in the corresponding approach file.
The approach file will be running on a 100GB file with the best approach we implemented
(optimization3) by default, only need to type following commands to run each task: “sbatch
1node1task.slurm”, “sbatch 1node8tasks.slurm”, “sbatch 2nodes8tasks.slurm”<p>

