conda create -n bert_ft  --yes  python=3.10
conda run    -n bert_ft  --live-stream conda install pip
conda run    -n bert_ft  --live-stream pip install -r requirements.txt
conda run    -n bert_ft  --live-stream python -m ipykernel install --user --name bert_ft --display-name "Python (bert_ft)"
