conda create -n llama3_ft  --yes  python=3.10
conda run    -n llama3_ft  --live-stream conda install pip
conda run    -n llama3_ft  --live-stream pip install -r requirements.txt
conda run    -n llama3_ft  --live-stream python -m ipykernel install --user --name llama3_ft --display-name "Python (lla
ma3_ft)"