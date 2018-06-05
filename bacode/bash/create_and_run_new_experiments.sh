# Add to pythonpath
PYTHONPATH="/Users/davidal/GoogleDrive/BachelorThesis/bacode/:/Users/davidal/GoogleDrive/BachelorThesis/"
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
    BASEPATH="/BachelorThesis/bacode/experiment_yamls/"
fi

delimiter="/"

initial_wd=$(pwd)

# Run the experiments ucb, and ucb_embed_benchmarkaber

config_path="$BASEPATH"
experiment_name="$dim"_"$model"

# Run the embedded ucb sample
config_path="$BASEPATH"johannes_configs
echo "Using config file:" $config_path$delimiter"new_algo.yaml"
febo create "new_algo" --config $config_path$delimiter"new_algo.yaml" --overwrite
# cd ..
febo run "new_algo"
febo plot "new_algo"
cd $initial_wd
