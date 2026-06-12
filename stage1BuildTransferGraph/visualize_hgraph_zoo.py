import argparse
import base64
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch


def load_payload(pt_path: str):
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def tensor_shape(x):
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return None


def edge_count(data, edge_type):
    return int(data[edge_type].edge_index.shape[1])


def node_count(data, node_type):
    return int(data[node_type].num_nodes)


def img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_summary_chart(data, out_dir: Path):
    rows = []

    for nt in data.node_types:
        rows.append({
            "category": f"node: {nt}",
            "count": node_count(data, nt),
        })

    for et in data.edge_types:
        src, rel, dst = et
        rows.append({
            "category": f"edge: {src}-{rel}->{dst}",
            "count": edge_count(data, et),
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=True)

    plt.figure(figsize=(11, max(4, 0.45 * len(df))))
    plt.barh(df["category"], df["count"])
    plt.xlabel("Count")
    plt.title("HGraph zoo: node and edge counts")
    plt.tight_layout()

    out_path = out_dir / "graph_summary.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    df.to_csv(out_dir / "graph_summary.csv", index=False)
    return out_path, df


def make_schema_diagram(out_dir: Path, has_xm0: bool):
    G = nx.DiGraph()

    G.add_node("TransferGraph\nrecords.csv", kind="source")
    G.add_node("model_config_dataset.csv", kind="source")
    G.add_node("dataset .npy\nDomainSimilarity", kind="source")
    G.add_node("lineage_records.csv", kind="source")

    G.add_node("GraphAttributesWith\nDomainSimilarity", kind="process")
    G.add_node("HGraph\nPyG HeteroData", kind="process")
    G.add_node("hgraph_zoo.pt", kind="artifact")

    G.add_node("model nodes", kind="node")
    G.add_node("dataset nodes", kind="node")
    G.add_node("model→dataset\naccuracy", kind="edge")
    G.add_node("model→dataset\ntransferability", kind="edge")
    G.add_node("dataset→dataset\nsimilarity", kind="edge")
    G.add_node("model→model\nlineage", kind="edge")

    if has_xm0:
        G.add_node("x_m^(0)\nname|desc|size|family", kind="feature")
    else:
        G.add_node("random model x\nsmoke mode", kind="feature")

    edges = [
        ("TransferGraph\nrecords.csv", "GraphAttributesWith\nDomainSimilarity"),
        ("model_config_dataset.csv", "GraphAttributesWith\nDomainSimilarity"),
        ("dataset .npy\nDomainSimilarity", "GraphAttributesWith\nDomainSimilarity"),
        ("lineage_records.csv", "GraphAttributesWith\nDomainSimilarity"),
        ("GraphAttributesWith\nDomainSimilarity", "model nodes"),
        ("GraphAttributesWith\nDomainSimilarity", "dataset nodes"),
        ("GraphAttributesWith\nDomainSimilarity", "model→dataset\naccuracy"),
        ("GraphAttributesWith\nDomainSimilarity", "model→dataset\ntransferability"),
        ("GraphAttributesWith\nDomainSimilarity", "dataset→dataset\nsimilarity"),
        ("GraphAttributesWith\nDomainSimilarity", "model→model\nlineage"),
        ("GraphAttributesWith\nDomainSimilarity", "x_m^(0)\nname|desc|size|family" if has_xm0 else "random model x\nsmoke mode"),
        ("model nodes", "HGraph\nPyG HeteroData"),
        ("dataset nodes", "HGraph\nPyG HeteroData"),
        ("model→dataset\naccuracy", "HGraph\nPyG HeteroData"),
        ("model→dataset\ntransferability", "HGraph\nPyG HeteroData"),
        ("dataset→dataset\nsimilarity", "HGraph\nPyG HeteroData"),
        ("model→model\nlineage", "HGraph\nPyG HeteroData"),
        ("x_m^(0)\nname|desc|size|family" if has_xm0 else "random model x\nsmoke mode", "HGraph\nPyG HeteroData"),
        ("HGraph\nPyG HeteroData", "hgraph_zoo.pt"),
    ]

    G.add_edges_from(edges)

    pos = {
        "TransferGraph\nrecords.csv": (-3, 2),
        "model_config_dataset.csv": (-3, 1),
        "dataset .npy\nDomainSimilarity": (-3, 0),
        "lineage_records.csv": (-3, -1),

        "GraphAttributesWith\nDomainSimilarity": (-1, 0.5),

        "model nodes": (1, 2),
        "dataset nodes": (1, 1.2),
        "model→dataset\naccuracy": (1, 0.4),
        "model→dataset\ntransferability": (1, -0.4),
        "dataset→dataset\nsimilarity": (1, -1.2),
        "model→model\nlineage": (1, -2),
        "x_m^(0)\nname|desc|size|family" if has_xm0 else "random model x\nsmoke mode": (1, 2.8),

        "HGraph\nPyG HeteroData": (3, 0.5),
        "hgraph_zoo.pt": (5, 0.5),
    }

    node_colors = []
    for n in G.nodes:
        kind = G.nodes[n]["kind"]
        if kind == "source":
            node_colors.append("#D9EAF7")
        elif kind == "process":
            node_colors.append("#FFE4B5")
        elif kind == "artifact":
            node_colors.append("#D7F5D0")
        elif kind == "feature":
            node_colors.append("#E6D9F7")
        else:
            node_colors.append("#F3F3F3")

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.4)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3600, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("Production graph-building pipeline")
    plt.axis("off")
    plt.tight_layout()

    out_path = out_dir / "graph_schema.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def short_name(name: str, max_len: int = 28):
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def make_sample_graph(data, unique_model_id, unique_dataset_id, out_dir: Path, top_models: int = 35):
    model_name = dict(zip(unique_model_id["mappedID"], unique_model_id["model"]))
    dataset_name = dict(zip(unique_dataset_id["mappedID"], unique_dataset_id["dataset"]))

    model_to_dataset_edge_types = [
        et for et in data.edge_types
        if et[0] == "model" and et[2] == "dataset"
    ]

    dataset_to_dataset_edge_types = [
        et for et in data.edge_types
        if et[0] == "dataset" and et[2] == "dataset"
    ]

    model_to_model_edge_types = [
        et for et in data.edge_types
        if et[0] == "model" and et[2] == "model"
    ]

    degree = Counter()

    for et in model_to_dataset_edge_types:
        edge_index = data[et].edge_index.cpu()
        for m in edge_index[0].tolist():
            degree[int(m)] += 1

    selected_models = {m for m, _ in degree.most_common(top_models)}
    selected_datasets = set(unique_dataset_id["mappedID"].astype(int).tolist())

    G = nx.Graph()

    for d in selected_datasets:
        label = short_name(dataset_name.get(d, f"dataset:{d}"))
        G.add_node(f"D:{d}", label=label, kind="dataset")

    for m in selected_models:
        label = short_name(model_name.get(m, f"model:{m}"))
        G.add_node(f"M:{m}", label=label, kind="model")

    edge_rows = []

    for et in model_to_dataset_edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index.cpu()
        for m, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            m = int(m)
            d = int(d)
            if m in selected_models and d in selected_datasets:
                G.add_edge(f"M:{m}", f"D:{d}", relation=rel)
                edge_rows.append({
                    "source": model_name.get(m, m),
                    "relation": rel,
                    "target": dataset_name.get(d, d),
                })

    # Add a small number of dataset-dataset similarity edges, so the graph still reads clearly.
    for et in dataset_to_dataset_edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index.cpu()
        limit = min(50, edge_index.shape[1])
        for i in range(limit):
            a = int(edge_index[0, i])
            b = int(edge_index[1, i])
            if a in selected_datasets and b in selected_datasets:
                G.add_edge(f"D:{a}", f"D:{b}", relation=rel)
                edge_rows.append({
                    "source": dataset_name.get(a, a),
                    "relation": rel,
                    "target": dataset_name.get(b, b),
                })

    # Add model-model lineage edges if present.
    for et in model_to_model_edge_types:
        src, rel, dst = et
        edge_index = data[et].edge_index.cpu()
        for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            a = int(a)
            b = int(b)
            if a in selected_models and b in selected_models:
                G.add_edge(f"M:{a}", f"M:{b}", relation=rel)
                edge_rows.append({
                    "source": model_name.get(a, a),
                    "relation": rel,
                    "target": model_name.get(b, b),
                })

    if G.number_of_nodes() == 0:
        raise RuntimeError("No nodes selected for sample graph.")

    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=120)

    node_colors = []
    node_sizes = []
    labels = {}

    for n, attrs in G.nodes(data=True):
        labels[n] = attrs["label"]
        if attrs["kind"] == "dataset":
            node_colors.append("#FFD166")
            node_sizes.append(1200)
        else:
            node_colors.append("#8ECAE6")
            node_sizes.append(550)

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.28, width=0.9)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.95,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title(f"HGraph zoo sample: all datasets + top {top_models} model nodes by graph degree")
    plt.axis("off")
    plt.tight_layout()

    out_path = out_dir / "graph_sample.png"
    plt.savefig(out_path, dpi=240)
    plt.close()

    pd.DataFrame(edge_rows).to_csv(out_dir / "sample_edges.csv", index=False)
    return out_path, pd.DataFrame(edge_rows)


def make_report(
    payload,
    data,
    summary_df,
    summary_img,
    schema_img,
    sample_img,
    unique_model_id,
    unique_dataset_id,
    out_dir: Path,
):
    args = payload.get("args", {})
    xm0_meta = payload.get("xm0_meta", None)
    contain_model_feature = args.get("contain_model_feature", None)

    model_preview = unique_model_id.head(12).to_html(index=False)
    dataset_preview = unique_dataset_id.head(12).to_html(index=False)
    summary_table = summary_df.sort_values("count", ascending=False).to_html(index=False)

    schema_b64 = img_to_base64(schema_img)
    summary_b64 = img_to_base64(summary_img)
    sample_b64 = img_to_base64(sample_img)

    xm0_html = "<p><b>xm0_meta:</b> not found. This file was likely built in smoke mode.</p>"
    if xm0_meta is not None:
        xm0_html = f"""
        <ul>
          <li><b>name_dim:</b> {xm0_meta.get("name_dim")}</li>
          <li><b>desc_dim:</b> {xm0_meta.get("desc_dim")}</li>
          <li><b>num_size_buckets:</b> {xm0_meta.get("num_size_buckets")}</li>
          <li><b>num_families:</b> {xm0_meta.get("num_families")}</li>
        </ul>
        """

    edge_type_rows = []
    for et in data.edge_types:
        edge_type_rows.append({
            "edge_type": str(et),
            "edge_index_shape": str(tuple(data[et].edge_index.shape)),
        })
    edge_type_table = pd.DataFrame(edge_type_rows).to_html(index=False)

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>HGraph Zoo Visualization Report</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 36px;
          line-height: 1.45;
          color: #222;
        }}
        h1, h2 {{
          margin-top: 28px;
        }}
        .card {{
          border: 1px solid #ddd;
          border-radius: 12px;
          padding: 18px;
          margin: 18px 0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        img {{
          max-width: 100%;
          border: 1px solid #eee;
          border-radius: 10px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin: 12px 0;
          font-size: 14px;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 6px 8px;
          text-align: left;
        }}
        th {{
          background: #f6f6f6;
        }}
        code {{
          background: #f3f3f3;
          padding: 2px 5px;
          border-radius: 4px;
        }}
      </style>
    </head>
    <body>
      <h1>HGraph Zoo Visualization Report</h1>

      <div class="card">
        <h2>1. What this file contains</h2>
        <p>
          This report visualizes <code>hgraph_zoo.pt</code>, a PyG HeteroData package
          containing model nodes, dataset nodes, model-dataset edges,
          dataset-dataset similarity edges, and optional model-model lineage edges.
        </p>
        <p><b>contain_model_feature:</b> {contain_model_feature}</p>
        {xm0_html}
      </div>

      <div class="card">
        <h2>2. Graph-building pipeline</h2>
        <img src="data:image/png;base64,{schema_b64}">
      </div>

      <div class="card">
        <h2>3. Node and edge counts</h2>
        <img src="data:image/png;base64,{summary_b64}">
        {summary_table}
      </div>

      <div class="card">
        <h2>4. Sample graph view</h2>
        <p>
          This view keeps all dataset nodes and selects the highest-degree model nodes,
          so the structure is readable during presentation.
        </p>
        <img src="data:image/png;base64,{sample_b64}">
      </div>

      <div class="card">
        <h2>5. Edge types inside HeteroData</h2>
        {edge_type_table}
      </div>

      <div class="card">
        <h2>6. First models and datasets</h2>
        <h3>Models</h3>
        {model_preview}
        <h3>Datasets</h3>
        {dataset_preview}
      </div>
    </body>
    </html>
    """

    out_path = out_dir / "graph_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default="hgraph_zoo.pt")
    parser.add_argument("--out_dir", default="viz_outputs")
    parser.add_argument("--top_models", type=int, default=35)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(args.pt)

    data = payload["data"]
    unique_model_id = payload["unique_model_id"]
    unique_dataset_id = payload["unique_dataset_id"]

    print("=== Loaded hgraph_zoo.pt ===")
    print(f"node_types: {data.node_types}")
    print(f"edge_types: {data.edge_types}")
    print(f"contain_model_feature: {payload.get('args', {}).get('contain_model_feature')}")
    print(f"xm0_meta exists: {'xm0_meta' in payload}")
    print()

    print("=== Nodes ===")
    for nt in data.node_types:
        print(f"{nt}: {data[nt].num_nodes}")
        for key, value in data[nt].items():
            if hasattr(value, "shape"):
                print(f"  {key}: {tuple(value.shape)}")
    print()

    print("=== Edges ===")
    for et in data.edge_types:
        print(f"{et}: {data[et].edge_index.shape[1]} edges")
        for key, value in data[et].items():
            if key != "edge_index" and hasattr(value, "shape"):
                print(f"  {key}: {tuple(value.shape)}")
    print()

    has_xm0 = "xm0_meta" in payload

    summary_img, summary_df = make_summary_chart(data, out_dir)
    schema_img = make_schema_diagram(out_dir, has_xm0=has_xm0)
    sample_img, sample_edges = make_sample_graph(
        data,
        unique_model_id,
        unique_dataset_id,
        out_dir,
        top_models=args.top_models,
    )

    report_path = make_report(
        payload=payload,
        data=data,
        summary_df=summary_df,
        summary_img=summary_img,
        schema_img=schema_img,
        sample_img=sample_img,
        unique_model_id=unique_model_id,
        unique_dataset_id=unique_dataset_id,
        out_dir=out_dir,
    )

    print("=== Wrote visualization outputs ===")
    print(f"summary chart: {summary_img}")
    print(f"schema chart:  {schema_img}")
    print(f"sample graph:   {sample_img}")
    print(f"report:         {report_path}")
    print(f"sample edges:   {out_dir / 'sample_edges.csv'}")


if __name__ == "__main__":
    main()