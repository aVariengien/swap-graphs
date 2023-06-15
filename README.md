# Swap graphs: Discovering the role of neural network components at scale

This librairy is an implemetation of input swap graphs described in this [post](https://www.lesswrong.com/posts/ZSYo97kcfwtFdpcwe/input-swap-graphs-discovering-the-role-of-neural-network). It is a tool to uncover the role of neural networks component by using causal interventions.

The librairy is built on top of [TransformerLens](https://github.com/neelnanda-io/TransformerLens). The code base is in very early stage and under active development. Feel free to contact me at `alexandre.variengien@gmail.com` if you have questions about the code or wants to make serious use of it.


I'd recommand starting with this [colab demo](https://colab.research.google.com/drive/1iZ0nB0aaQSkJRyfAP4DDLKLsbUTfP_V5?usp=sharing). For more advanced example that uses swap graphs to craft validation experiments, you can explore the `tests` directory.

### Install

`pip install git+https://github.com/aVariengien/swap-graphs.git`

### Scripts

We also provide scripts to handle swap graphs at scale.
* `compute_sgraphs.py` is used to compute a swap graphs for every component at a given position (often the last position in a sequence).
* `plot_semantic_maps.py` uses the fiels created by `compute_sgraphs.py` to create the semantic maps visualisation.
* `sgraph_causal_scrubbing.py` runs causal scrubbing experiments where all components up to layer L are scrubbed.
* `targetted_rewrite.py` (only for the IOI dataset) runs targetted rewrite experiments for the senders and extended name mover heads.

