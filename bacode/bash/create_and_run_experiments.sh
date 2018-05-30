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

FOLDERNAMES=("high5_low1" "high5_low2" "high10_low5") #"high2_low1"
EXPERIMENTCONFIGS=("stdd" "trip")

delimiter="/"

initial_wd=$(pwd)

for dim in "${FOLDERNAMES[@]}"
do
	for model in "${EXPERIMENTCONFIGS[@]}"
    do
        config_path="$BASEPATH$dim$delimiter$model"
        experiment_name="$dim"_"$model"
        echo "Using config file: $config_path"
        echo "Creating experiment: "
        febo create "$dim"_"$model" --config $config_path".yaml" --overwrite
        cd ..
        febo run "$dim"_"$model"
        cd $initial_wd
    done
done

# Now run each experiment individually