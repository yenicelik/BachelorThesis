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
fi

delimiter="/"

initial_wd=$(pwd)

# Run the experiments ucb, and ucb_embed_benchmarkaber

config_path="$BASEPATH"
experiment_name="$dim"_"$model"


# Run the standard ucb sample
# config_path="$BASEPATH"johannes_configs
# echo "Using config file:" $config_path$delimiter"ucb.yaml"
# febo create "johannes_ucb" --config $config_path$delimiter"ucb.yaml" --overwrite
# # cd ..
# febo run "johannes_ucb"
# febo plot "johannes_ucb"
# cd $initial_wd


# Run the embedded ucb sample
config_path="$BASEPATH"johannes_configs
echo "Using config file:" $config_path$delimiter"ucb_embed_benchmark.yaml"
febo create jucb --config $config_path$delimiter"ucb_embed_benchmark.yaml" --overwrite
# cd ..
febo run jucb

printf "\n\n\n\n\nDONE RUNNING. NOW VISUALIZING\n\n\n\n\n\n\n\n"

febo plot jucb
cd $initial_wd
