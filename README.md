# Pub2FedIE

Code for *Public biomedical adapter preparation improves federated clinical information extraction with less local training and communication*. The tasks are named-entity recognition (NER) and conditional relation extraction (RE).

## Install

From the repository root, use a Python 3.10 environment with CUDA-enabled PyTorch (the paper used PyTorch 2.5.1 with CUDA 12.1):

```bash
python -m pip install -r requirements.txt
```

## Train and evaluate

The example prepares one public adapter, compares continuation with residual adaptation on the same clients, and evaluates both methods. Obtain a local Qwen or Llama snapshot from its provider. Choose your own data paths, steps, and seeds, or see [study.json](study.json) for the paper’s model revisions and settings.

<details>
<summary>Input files and data formats</summary>

Pass a JSON file to `--clinical` that maps each task, split, and site to its data file. Choose your own site IDs. Relative paths start in the map's directory:

```json
{
  "NER": {
    "train": {"North": "annotations/north-train.json", "South": "/data/south/training.json"},
    "test":  {"North": "annotations/north-test.json", "South": "/data/south/test.json", "External": "held-out.json"}
  }
}
```

Each file is a nonempty JSON array. For NER, provide aligned `tokens` and BIO `tags`. For conditional RE, mark one primary entity in `input` and its modifiers in `output`. The [configuration](study.json) lists the labels for each task. Put external evaluation sites under `test` only.

The paper uses Hu et al.'s splits, with i2b2 held out for evaluation. [Kiwi-LLaMA](https://github.com/BIDS-Xu-Lab/Kiwi-LLaMA) provides the annotated MTSamples data and the entity text and character offsets for MIMIC-III and the reannotated 2010 i2b2 subset. Request the underlying notes from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) and the [n2c2 Data Portal](https://n2c2.dbmi.hms.harvard.edu/data-sets), respectively. UT Physicians records are not publicly available.

For public preparation, pass the paths to the six corpora via `--corpora`:

```json
{
  "bc5cdr": "CDR_Data.zip",
  "ncbi": "NCBI_corpus.zip",
  "drugprot": {"abstracts": "dp/text.tsv", "entities": "dp/entities.tsv", "relations": "dp/relations.tsv"},
  "biored": "biored/Train.BioC.JSON",
  "ddi": ["ddi/train/DrugBank", "ddi/train/MedLine"],
  "e3c_ner": "e3c/english-ner-datasetdict",
  "e3c_re": "e3c/english-training-relations.json"
}
```

Use the training files for each corpus. The ZIP readers use the providers' original member names; each DDI directory contains XML files, and the E3C NER directory is a Hugging Face DatasetDict saved to disk. The original study directory layouts still work as an alternative to path maps.

</details>

### Run a paired NER comparison

Set `MODEL` to the local Qwen snapshot, `PUBLIC` and `CLINICAL` to your path maps, and `RUN` to an output directory outside the repository. Run the example from the repository root. Both methods start from the same public adapter and see the same private training order.

```bash
set -eu
: "${MODEL:?set MODEL}" "${PUBLIC:?set PUBLIC}" "${CLINICAL:?set CLINICAL}" "${RUN:?set RUN}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
block=7
seed=13
python -m pub2fedie.prepare --model Qwen --snapshot "$MODEL" --corpora "$PUBLIC" \
  --preparation SFT --block "$block" --out "$RUN/public-$block"

for method in SFT-Cont SFT-Res; do
  fit="$RUN/fits/$method/$block/$seed"
  python -m pub2fedie.adapt --model Qwen --snapshot "$MODEL" --clinical "$CLINICAL" \
    --task NER --method "$method" --public "$RUN/public-$block/public.pt" \
    --block "$block" --seed "$seed" --steps 100 --rounds 1 --out "$fit"
  for site in North South; do
    python -m pub2fedie.evaluate --snapshot "$MODEL" --clinical "$CLINICAL" --site "$site" \
      --checkpoint "$fit/r1_global.pt" --out "$RUN/evals/$method/$block/$seed/$site"
  done
done

python -m pub2fedie.results --left "$RUN/evals/SFT-Cont" --right "$RUN/evals/SFT-Res"
```

This gives one paired F1 difference. For a confidence interval, repeat both methods with additional paired seeds or blocks, then rerun the last command. Reuse the public adapter when changing only the private seed. For an external comparison, evaluate the external site for every pair and pass its ID with `--domain`.

Set public training length with `--updates`. Private `--steps` is the total optimizer steps per client; choose a positive total divisible by the number of rounds. On a non-Fed one-round trajectory, `--save-steps 10 50` saves local and global snapshots for evaluation while each client continues its own trajectory.

<details>
<summary>Other methods and aggregation options</summary>

| Method | What changes |
|---|---|
| No-Public | Skip public preparation and omit `--public`; fit fresh client adapters. |
| SFT-Cont / Meta-Cont | Each client updates its copy of the public adapter; the server aggregates the updated adapters. |
| SFT-Res / Meta-Res | Freeze the public adapter and fit a new residual. For Meta-Res and Meta-Cont, use `--preparation Meta`. |
| Meta with simulated continuation | During public preparation, simulated clients update copies of the shared adapter. The query loss updates the initialization through rank-16 aggregation. Clinical clients then continue the prepared adapter (Qwen NER, Table S3). |
| Fed-MedLoRA+ | Fit fresh adapters; weight their A/B factors using the [released validation-loss rule](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/tree/f8d4821021303463990d76b9844f38003296fafa). |

To run the Table S3 ablation, use the same name for preparation and adaptation:

```bash
method="Meta with simulated continuation"
python -m pub2fedie.prepare --model Qwen --snapshot "$MODEL" --corpora "$PUBLIC" \
  --preparation "$method" --block "$block" --out "$RUN/simulated-public-$block"
python -m pub2fedie.adapt --model Qwen --snapshot "$MODEL" --clinical "$CLINICAL" \
  --task NER --method "$method" --public "$RUN/simulated-public-$block/public.pt" \
  --block "$block" --seed "$seed" --rounds 1 --steps 480 --save-steps 30 60 120 240 \
  --out "$RUN/fits/$method/$block/$seed"
```

Evaluate each budget with `--checkpoint`: use `step30/r1_global.pt` (likewise for 60, 120 and 240) or `r1_global.pt` for 480, relative to that fit's output directory.

Fed validation uses distinct training examples held out from client fitting. Set the number per site with `--panel-size N` (default 32), or pass a site-to-index map with `--panel indices.json` (zero-based train-row indices). By default, the panel excludes examples used in that run. To reproduce the paper's shared panel across budgets, use `--panel-exclude-steps 480`; use 60 for the original RE60 fixed-client ablation.

Training already produces a global adapter. To compare aggregation rules on the **same saved one-round SFT-Cont clients**, without retraining (Tables S10–S11):

```bash
python -m pub2fedie.adapt --reaggregate "$RUN/fits/SFT-Cont/$block/$seed" \
  --clinical "$CLINICAL" --snapshot "$MODEL" --rules A B C D --out "$RUN/rules"
```

A/B average effective updates; C/D average LoRA factors. A/C weight clients equally; B/D weight them by validation loss. Choose just the rules you need (e.g., `--rules A D`); A/C alone run on CPU without `--snapshot`. Pass each output (e.g., `$RUN/rules/A.pt`) to `pub2fedie.evaluate --checkpoint`. For own-local evaluation, pass the checkpoint for the corresponding site instead.

</details>

## Compare evaluation outputs on CPU

For the statistics alone, install NumPy and SciPy (`python -m pip install numpy==1.26.4 scipy==1.14.0`):

```bash
python -m pub2fedie.results --left path/to/cont-evals --right path/to/res-evals
```

Omit `--right` for one method's mean F1 and 95% confidence interval (Tables S2/S4). With two methods, the output also gives the paired difference in F1 points and its 95% interval.

<details>
<summary>Pairing and confidence intervals</summary>

The CPU comparison reads the `counts.json` produced by evaluation. The two arms must contain the same observed (initialization block, private seed) pairs for every requested site, using identical test splits. The default averages site F1s across the training cohort; use `--sites North South` to name a cohort explicitly or `--domain External` to score one site. Use `--label` to restrict scoring to one entity or modifier.

Blocks receive equal weight, as do paired runs within each block. The 95% percentile interval uses 20,000 bootstrap draws: resample blocks, then paired private runs within each sampled block, keeping the test sets fixed. With one block, the interval reflects variation across its private runs. A single paired run gives a point estimate only. For the paper's Qwen NER480 and RE60 fixed-client contrasts in S11, use `--bootstrap-seed 2026101914`.

For Table S17, the same comparison reports the mean difference within each initialization block, how many are positive, and a t interval across block means (two degrees of freedom for the paper's three blocks). This sensitivity interval needs at least two blocks.

</details>
