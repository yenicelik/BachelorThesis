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
    BASEPATH="/Users/davidal/GoogleDrive/BachelorThesis/code/experiment_yamls/"
else
    BASEPATH="/BachelorThesis/code/experiment_yamls/"
fi

FOLDERNAMES=("high2_low1" "high5_low1" "high5_low2" "high10_low5")
EXPERIMENTCONFIGS=("stdd.yaml" "trip.yaml")

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
        febo create $dim --config "$config_path" --overwrite
        cd ..
        febo run $dim
        cd $initial_wd
    done
done

# Now run each experiment individually