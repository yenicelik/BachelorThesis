# Add to pythonpath
PYTHONPATH="/Users/davidal/GoogleDrive/BachelorThesis/bacode/:/Users/davidal/GoogleDrive/BachelorThesis/:/home/yedavid/BachelorThesis/bacode/:/home/yedavid/BachelorThesis/"
export PYTHONPATH

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo ${machine}

if [[ $machine == *"Mac"* ]];
then
    BASEPATH="/Users/davidal/GoogleDrive/BachelorThesis/bacode/experiment_yamls/"
else
    BASEPATH="/home/yedavid/BachelorThesis/bacode/experiment_yamls/"
    cd /home/yedavid/BachelorThesis/bacode/
fi


python tripathy/experiments/visualize_with_predictions/00_main_visualize_predictions.py