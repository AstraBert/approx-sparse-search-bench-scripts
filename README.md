# Scripts for Benchmarking Approximate Sparse Search In Qdrant

The `scripts` folder contains python programs to benchmark approximate sparse search in Qdrantl, for both `query_dims` and `query_mass`, and to estimate what cutting does to a query in terms of information loss.

Specifically:

- `scripts/splade/query_cut_bench*.py`: benchmark latency and quality deriving from the use of `query_mass`/`query_dims` on [this SPLADE dataset](https://huggingface.co/datasets/tuskanny/seismic-nq-splade) derived from [`BeIR/nq`](https://huggingface.co/datasets/BeIR/nq). The dataset is expected to be on disk.
- `scripts/slideon/query_cut_bench*.py`: benchmark latency and and quality deriving from the use of `query_mass`/`query_dims` on datasets that have the same format as [this SlideOn dataset](https://huggingface.co/datasets/lightonai/SlideOn-sparse-vectors), also derived from `nq`. The dataset can be loaded from HuggingFace or from local disk.
- `scripts/query_cut_terms.py` measures the effect of cutting query dimensions for different vocabulary-based sparse neural models. Specifically, it measures how many original and expansion terms where lost due to the cut, as well as how much weight was lost.

All these scripts are standalone CLIs that can be executed directly without a virtual environment, as long as [uv](https://astral.sh/docs/uv) is installed. The first time you run the script, it installs the needed dependencies (for some heavy ones, it can take some time).
