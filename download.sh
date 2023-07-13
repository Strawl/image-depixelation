!rm -r ./image*

git clone https://github.com/Strawl/image-depixelation.git
pip install -r ./image-depixelation/requirements.txt
!chmod +x ./image-depixelation/download.sh

rm -r ./tmp

# Download the files
curl -o training.zip https://cloud.ml.jku.at/s/ARpJJGeoCnojqsJ/download/training.zip
curl -o test_set.pkl https://cloud.ml.jku.at/s/WrGXgMFwwEd2opt/download/test_set.pkl

# Create directories
mkdir -p ./tmp/images/
mkdir -p ./tmp/

# Move test_set.pkl to its place
mv test_set.pkl ./tmp/test_set.pkl

# Extract zip file to the specified directory
unzip training.zip -d ./tmp/images/

# Move contents from ./tmp/images/training/* to ./tmp/images/ and then remove the empty directory
mv ./tmp/images/training/* ./tmp/images/
rmdir ./tmp/images/training/

mv training.zip ./tmp/
