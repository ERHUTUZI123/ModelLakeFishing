python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

cd stage1BuildTransferGraph

python test_hgraph_minimal.py

python build_graph.py --contain_model_feature True

python visualize_hgraph_zoo.py --pt hgraph_zoo.pt --top_models 35