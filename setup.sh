echo Downloading the midi dataset
curl -LO  https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
unzip maestro-v2.0.0-midi.zip
echo Creating virtual environment
conda create -n midi-rl python=3.7
conda activate midi-rl
echo Installing dependencies
pip install -r requirements.txt
source activate midi-rl
