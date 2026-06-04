python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cd stage1BuildTransferGraph
python test_hgraph_minimal.py