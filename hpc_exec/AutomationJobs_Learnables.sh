# execute in terminal with & linebreak disown that it runs independently from the terminal and ssh connection

sbatch AutoML_Learnables.sh
echo "submitted first job"

echo "wait 12h now"
sleep $((12 * 60 * 60)) #sec * min * hours / in seconds
echo "waited 12h - submitting second job now with sbatch"

sbatch AutoML_Learnables.sh
echo "submitted second job"

echo "wait 12h now"
sleep $((12 * 60 * 60)) #sec * min * hours / in seconds
echo "waited 12h - submitting third job now with sbatch"

sbatch AutoML_Learnables.sh
echo "submitted third job"

echo "wait 12h now"
sleep $((12 * 60 * 60)) #sec * min * hours / in seconds
echo "waited 12h - submitting fourth job now with sbatch"

sbatch AutoML_Learnables.sh
echo "submitted fourth job"