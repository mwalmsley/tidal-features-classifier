# Running on AWS

### Get data from S3

aws configure
(enter credentials)

aws s3 sync s3://galaxy-zoo/tidalclassifier/subjects/static_processed/512 ~/subjects/static_processed/512
aws s3 sync s3://galaxy-zoo/tidalclassifier/tables/training tables/training

### Add Github SSH key (previously added to s3)
<!-- acceptable security risk given the context -->
aws s3 sync s3://mikewalmsley/deprecated/id_rsa ~/.ssh/
<!-- Remove Mac OSX config, will take out of S3 soon -->
rm ~/.ssh/config
eval "$(ssh-agent -s)"
chmod 400 ~/.ssh/github
ssh-add ~/.ssh/github
git clone git@github.com:RustyPanda/tidal-features-classifier.git tidalclassifier

### Set Up Python Environment
source activate tensorflow_p36
pip install -e tidalclassifier
pip install -r tidalclassifier/requirements.txt  # with tensorflow and keras commented out
pip install photutils --no-deps
pip install seaborn --upgrade

### Run
cd tidalclassifier
mkdir results
mkdir results/cnn_runs

(run a brief test to make sure everything works okay)
python tidalclassifier/cnn/individual_cnn/run_meta.py --name=aws_test --results_dir=results/cnn_runs --cv_mode=random --aws --test_mode

(your new run here)
