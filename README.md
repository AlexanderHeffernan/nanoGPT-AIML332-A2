# AIML332 Assignment 2

This repository contains my submission for AIML332 Assignment 2. I have forked and extended [nanoGPT](https://github.com/karpathy/nanoGPT) to fine-tune GPT-2 on a specialised cricket rules dataset, and implemented several extensions for model evaluation and analysis. Please see [report.md](report.md) for a detailed write-up.

## Setup

Before running any code, install the required Python packages:

```bash
pip install torch tiktoken matplotlib
```

If you are on a Mac with Apple Silicon, you may want to use the `mps` device for faster training/inference. Else, use `cuda` if you have an NVIDIA GPU, or `cpu` otherwise.

### Download and Setup the Fine-Tuned Model Checkpoint

The fine-tuned cricket model checkpoint (`ckpt.pt`) is too large for GitHub version control. Please download it from the [GitHub Release page](https://github.com/AlexanderHeffernan/nanoGPT-AIML332-A2/releases/tag/1.0) and place it in the `out-cricketrules` directory.

1. Got to: [GitHub Release](https://github.com/AlexanderHeffernan/nanoGPT-AIML332-A2/releases/tag/1.0)
2. Download `ckpt.pt` from the Assets section.
3. Move `ckpt.pt` to the `out-cricketrules` directory:
```bash
mv ~/Downloads/ckpt.pt out-cricketrules/
```
Or manually drag and drop it into the `out-cricketrules` folder.

Now you can run all commands in this README as described.

## How to Run the Programs

### 1. Visualise Token Probabilities (`sample.py --show_probs`) (1.1)

To see how the model decides on the next token, use the `--show_probs` flag with `sample.py`. This will plot the top 10 token probabilities at each generation step.

```bash
python sample.py --init_from=gpt2 --start="I live in" --num_samples=1 --max_new_tokens=20 --device=mps --show_probs=True
```

A Matplotlib bar chart will appear after each token is generated, showing the model's confidence.

### 2. Sequence Probability (1.2)

The model now prints the probability assigned to the generated sequence after each sample. This is shown automatically in the output of `sample.py`.

```
Sequence probability: 1.2345e-04
```

This is calculated by multiplying the probabilities of each generated token (see [report.md](report.md) for details).

### 3. Probability of a Fixed Response (`sample.py --fixed_response`) (1.3)

To compute the probability that the model assigns to a specific, fixed response after a prompt, use the `--fixed_response` flag:

```bash
python sample.py --init_from=gpt2 --start="What is the capital of France?" --fixed_response="The capital of France is Paris." --num_samples-1 --device=mps
```

### 4. Evaluation Harness (`eval.py`) (2.1)

To evaluate the gpt2 model on a set of prompt-response pairs ([eval_data](eval_data.json)), run:

```bash
python eval.py --init_from=gpt2 --device=mps
```

### 5. Evaluate the Cricket Fine-Tuned Model (2.3)

To evaluate the fine-tuned cricket model, make sure you set the correct output directory:

```bash
python eval.py --init_from=resume --out_dir=out-cricketrules --device=mps
```

### 6. Beam Search Decoding Extension (3.2)

I implemented **beam search decoding** as an extension in [`model.py`](model.py) and added support for it in [`sample.py`](sample.py). Beam search keeps track of the top-k most probable sequences (beams) at each generation step, expanding the pruning them to select the most likely overall output.

To use beam search, run:

```bash
python sample.py --init_from=gpt2 --start="I live in" --num_samples=1 --max_new_tokens=20 --device=mps --beam_search=True --beam_width=3
```

## Reference

For further details on the implementation and results, please refer to [report.md](report.md).

---
**Node:** This project is for educational purposes and builds on [nanoGPT](https://github.com/karpathy/nanoGPT).