# Add to pythonpath
PYTHONPATH="/Users/david/GoogleDrive/BachelorThesis/bacode/:/Users/david/GoogleDrive/BachelorThesis/:/home/yedavid/BachelorThesis/bacode/:/home/yedavid/BachelorThesis/"
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
    BASEPATH="/Users/david/GoogleDrive/BachelorThesis/bacode/experiment_yamls/"
else
    BASEPATH="/home/yedavid/BachelorThesis/bacode/experiment_yamls/"
fi

delimiter="/"

initial_wd=$(pwd)

# Run the experiments ucb, and ucb_embed_benchmarkaber

config_path="$BASEPATH"
experiment_name="$dim"_"$model"

# Run the embedded ucb sample
config_path="$BASEPATH"johannes_configs
echo "Using config file:" $config_path$delimiter"01_parabola_config.yaml"
febo create parabola_all --config $config_path$delimiter"01_parabola_config.yaml" --overwrite
# cd ..
febo run parabola_all

printf "\n\n\n\n\nDONE RUNNING. NOW VISUALIZING\n\n\n\n\n\n\n\n"

febo plot parabola_all
cd $initial_wd
