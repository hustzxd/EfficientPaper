import os
import sys

import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_paper_pb2 as eppb

sys.path.append("./")



PUBLISH_COLOR = {
    # AI/ML Conferences (warm colors)
    "AAAI": "FF4500",  # Orange Red
    "ICLR": "FF6B6B",  # Coral Pink
    "ICML": "FF8C00",  # Dark Orange
    "NeurIPS": "FF1493",  # Deep Pink
    # NLP Conferences (cool colors)
    "ACL": "4169E1",  # Royal Blue
    "COLM": "6495ED",  # Cornflower Blue
    "Coling": "1E90FF",  # Dodger Blue
    "ENLSP": "00BFFF",  # Deep Sky Blue
    "TACL": "87CEEB",  # Sky Blue
    # CV Conferences (green variants)
    "CVPR": "2E8B57",  # Sea Green
    "ECCV": "3CB371",  # Medium Sea Green
    # Systems Conferences (purple variants)
    "ASPLOS": "9370DB",  # Medium Purple
    "SOSP": "8A2BE2",  # Blue Violet
    "ISCA": "9932CC",  # Dark Orchid
    "MICRO": "BA55D3",  # Medium Orchid
    "MLSys": "DDA0DD",  # Plum
    # Architecture/Hardware (red variants)
    "ATC": "DC143C",  # Crimson
    "DATE": "B22222",  # Fire Brick
    "SC": "CD5C5C",  # Indian Red
    "TC": "F08080",  # Light Coral
    "VLDB": "A52A2A",  # Brown
    "VLSI": "8B0000",  # Dark Red
    # Journals (teal variants)
    "JMLR": "008B8B",  # Dark Cyan
    "TMLR": "20B2AA",  # Light Sea Green
    "Neuromorphic Computing and Engineering": "40E0D0",  # Turquoise
    # Others (neutral colors)
    "arXiv": "1E88E5",  # Bright Blue (Material Design Blue)
    "AutoML Workshop": "A9A9A9",  # Dark Gray
    "Blog": "696969",  # Dim Gray
    "github": "2F4F4F",  # Dark Slate Gray
}


def readMeta():
    pinfos = []
    for year in os.listdir("./meta"):
        if year == "search":
            continue
        for f in os.listdir(f"./meta/{year}"):
            # todo: add year dir
            pinfo = eppb.PaperInfo()
            try:
                with open(os.path.join("./meta", year, f), "r") as rf:
                    pb.text_format.Merge(rf.read(), pinfo)
                pinfos.append((pinfo, f))
            except:
                print("read error in {}".format(f))

    return pinfos


word_pb2str = {
    eppb.Keyword.Word.none: "None",
    eppb.Keyword.Word.sparse_pruning: "Sparse/Pruning",
    eppb.Keyword.Word.weight_sparsity: "Sparsity (Weight)",
    eppb.Keyword.Word.activation_sparsity: "Sparsity (Activation)",
    eppb.Keyword.Word.attention_sparsity: "Sparsity (Attention)",
    eppb.Keyword.Word.structured_sparsity: "Sparsity (Structured)",

    eppb.Keyword.Word.quantization: "Quantization",

    eppb.Keyword.Word.kv_cache_quant: "Quantization (KV Cache)",
    eppb.Keyword.Word.kv_cache_sparse: "Sparse/Eviction (KV Cache)",
    eppb.Keyword.Word.kv_cache_management: "KV Cache Management",

    eppb.Keyword.Word.survey: "Survey",
    eppb.Keyword.Word.low_rank: "Low Rank Decomposition",
    eppb.Keyword.Word.fusion: "Layer Fusion (Reduce IO)",
    eppb.Keyword.Word.tool: "Tool",
    eppb.Keyword.Word.efficient_training: "Efficient Training",
    eppb.Keyword.Word.structure_design: "Network Structure Design",
    eppb.Keyword.Word.deployment: "LLM Deployment",
    eppb.Keyword.Word.overlap: "Comm-Comp Overlap",
    eppb.Keyword.Word.speculative_decoding: "Speculative Decoding",
    eppb.Keyword.Word.performance_modeling: "Performance Modeling",
    eppb.Keyword.Word.benchmark: "Benchmark",

    eppb.Keyword.Word.kernel_generation: "Kernel Generation"
}

for k, v in word_pb2str.items():
    word_pb2str[k] = f"{k-1:02}-{v}"


