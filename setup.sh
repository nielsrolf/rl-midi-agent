echo Downloading the midi dataset
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
unzip maestro-v2.0.0-midi.zip
echo Creating virtual environment
conda create -n midi-rl python=3.6
conda activate midi-rl
conda env config vars set PYTHONPATH=$(pwd):$PYTHONPATH
echo Installing dependencies
pip install -r requirements.txt