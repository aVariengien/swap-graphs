# Swap graphs: Discovering the role of neural network components at scale

This library is an implementation of input swap graphs described in this [post](https://www.lesswrong.com/posts/ZSYo97kcfwtFdpcwe/input-swap-graphs-discovering-the-role-of-neural-network). It is a tool to uncover the role of neural networks component by using causal interventions.

The library is built on top of [TransformerLens](https://github.com/neelnanda-io/TransformerLens). The code base is in a very early stage and under active development. Feel free to contact me at `alexandre.variengien@gmail.com` if you have questions about the code or want to make serious use of it.

I'd recommend starting with this [colab demo](https://colab.research.google.com/drive/1iZ0nB0aaQSkJRyfAP4DDLKLsbUTfP_V5?usp=sharing). For a more advanced example that uses swap graphs to craft validation experiments, you can explore the `tests` directory.

You can also check out the [nanoQA demo](https://colab.research.google.com/drive/15W2CSB0y77flpB3CtVUplvcy_F_bRSjw?usp=sharing) that demonstrates how to use swap graphs to investigate how GPT-2 small answer questions in-context!

### Install

`pip install git+https://github.com/aVariengien/swap-graphs.git`

### Scripts

We also provide scripts to handle swap graphs at scale.
* `compute_sgraphs.py` is used to compute a swap graph for every component at a given position (often the last position in a sequence).
* `plot_semantic_maps.py` uses the fiels created by `compute_sgraphs.py` to create the semantic maps visualisation.
* `sgraph_causal_scrubbing.py` runs causal scrubbing experiments where all components up to layer L are scrubbed.
* `targetted_rewrite.py` (only for the IOI dataset) runs targetted rewrite experiments for the senders and extended name mover heads.

