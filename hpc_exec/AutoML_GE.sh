#!/usr/bin/zsh

#SBATCH --job-name=GE_4nd            # Job name
#SBATCH --output=output_%j.log              # Output log file, %j will be replaced by job ID
#SBATCH --error=error_%j.log                # Error log file, %j will be replaced by job ID

#SBATCH --ntasks-per-node=96                # Number of tasks 
#SBATCH --cpus-per-task=1                   # Number of cpus per taks
#SBATCH --time=12:00:00                     # Time limit hrs:min:sec (12 hours max for our project)
#SBATCH --nodes=4
##SBATCH --exclusive

#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0

#SBATCH --account=rwth1479
#SBATCH --partition=c23mm

##SBATCH --account=lect0135
##SBATCH --partition=c23g

##sacct --format="JobID,JobName%30" #display extended job name 

## uncomment for GPU usage
#module load CUDA/12.2.2
#echo; export; echo; nvidia-smi; echo

# Run the Python code inside the Apptainer container
echo "starting first srun"
#srun --wait=0 nsys profile --gpu-metrics-device=all apptainer exec --bind $EBROOTCUDA --nv stecher_apptainer.sif python ClusterScript_AutoML_GE_load_copy.py #nsys=profiling
srun --wait=0 apptainer exec /rwthfs/rz/cluster/work/TIMID/AutoML/apptainer/lambda_stack_2204_aptdef.sif python3 execute_study_GE.py
#--nv 

# clean up dirs due to space reasons
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/model_dir_BO_GE
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/log_dir_BO_GE

echo "starting second srun"
sleep 15
srun --wait=0 apptainer exec /rwthfs/rz/cluster/work/TIMID/AutoML/apptainer/lambda_stack_2204_aptdef.sif python3 execute_study_GE.py

# clean up dirs due to space reasons
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/model_dir_BO_GE
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/log_dir_BO_GE

echo "starting third srun"
sleep 15
srun --wait=0 apptainer exec /rwthfs/rz/cluster/work/TIMID/AutoML/apptainer/lambda_stack_2204_aptdef.sif python3 execute_study_GE.py

# clean up dirs due to space reasons
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/model_dir_BO_GE
rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/log_dir_BO_GE

# echo "starting fourth srun"
# sleep 15
# srun --wait=0 apptainer exec /rwthfs/rz/cluster/work/TIMID/AutoML/apptainer/lambda_stack_2204_aptdef.sif python3 execute_study_GE.py

# # clean up dirs due to space reasons
# rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/model_dir_BO_GE
# rm -rf /rwthfs/rz/cluster/home/TIMID/AutoML/experiment_logs/log_dir_BO_GE