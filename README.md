python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

cd stage1BuildTransferGraph

python test_hgraph_minimal.py

REAL

python build_graph.py --contain_model_feature True --out hgraph_zoo_xm0.pt

python visualize_hgraph.py --pt hgraph_zoo_xm0.pt --out hgraph_zoo_xm0_viz.png

NOT REAL

python build_graph.py --contain_model_feature False --out hgraph_zoo.pt

python visualize_hgraph.py --pt hgraph_zoo.pt --out hgraph_zoo_viz.png