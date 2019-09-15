
echo "Downloading checkpoints..."
mkdir checkpoints
cd checkpoints
wget http://rpg.ifi.uzh.ch/data/unsupervised_detection_models.zip
unzip unsupervised_detection_models.zip
rm unsupervised_detection_models.zip
cd ..

echo "Creating tf-1.10 env..."
virtualenv -p python3 tf-1.10
source tf-1.10/bin/activate
pip install tensorflow-gpu==1.10

echo "Installing dependencies..."
pip install -r requirements.txt

echo "tf-1.10 environment is ready to be used!"