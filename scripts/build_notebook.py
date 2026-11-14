"""Generates notebooks/colab_finetune.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Part II — Fine-tuning the ticket classifiers (Google Colab)\n"
       "AI Service Desk Copilot · Group 60 · AIMLZG549 Assignment II\n\n"
       "**Runtime → Change runtime type → T4 GPU** before you start. The whole notebook takes about 45–75 minutes.\n\n"
       "What it produces (all needed for the report):\n"
       "1. Clean train / validation / test splits and EDA charts\n"
       "2. Two baselines: TF-IDF + logistic regression, and a zero-shot LLM\n"
       "3. Fine-tuned DeBERTa-v3 **queue** classifier and **priority** classifier, logged to MLflow\n"
       "4. Confusion matrices, training curves, before/after comparison tables\n"
       "5. Models pushed to the Hugging Face Hub (used by the app) and a zip of all results"),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## 1. Get the code\nUse **option A** after pushing the repo to GitHub, or **option B** to upload the zip."),
    code("# Option A: clone from GitHub (edit the URL if your repo name differs)\n"
         "REPO_URL = \"https://github.com/prashob-narendran/servicedesk-copilot.git\"  #@param {type:\"string\"}\n"
         "!git clone -q $REPO_URL servicedesk-copilot"),
    code("# Option B: upload servicedesk-copilot.zip instead (skip if option A worked)\n"
         "# from google.colab import files\n"
         "# files.upload()\n"
         "# !unzip -q servicedesk-copilot.zip"),
    code("%cd /content/servicedesk-copilot\n!pip install -q -r requirements-train.txt"),
    md("## 2. Secrets\nAdd these in Colab's **🔑 Secrets** panel (left sidebar) and enable notebook access:\n"
       "- `OPENAI_API_KEY` — for the zero-shot baseline\n- `HF_TOKEN` — a Hugging Face token with **write** access\n"
       "- set `HF_USER` below to your Hugging Face username"),
    code("import os\nfrom google.colab import userdata\n"
         "os.environ[\"OPENAI_API_KEY\"] = userdata.get(\"OPENAI_API_KEY\")\n"
         "os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")\n"
         "HF_USER = \"your-hf-username\"  #@param {type:\"string\"}\n"
         "from huggingface_hub import login\nlogin(token=os.environ[\"HF_TOKEN\"])"),
    md("## 3. Prepare the data\nEnglish tickets only, empty and duplicate tickets removed, 80/10/10 split stratified by queue "
       "(seed 42). If loading fails, the error lists the CSV files in the dataset repo; re-run with `--hub-file <name>`."),
    code("!python -m data.prepare"),
    code("from IPython.display import Image, display\n"
         "for f in [\"queue_distribution\", \"priority_distribution\", \"ticket_length\"]:\n"
         "    display(Image(f\"reports/figures/{f}.png\", width=650))"),
    md("## 4. Baselines (the 'before' numbers)"),
    code("!python -m classifier.baselines --label queue --zero-shot\n!python -m classifier.baselines --label priority"),
    md("## 5. Fine-tune the queue classifier\nAbout 20–35 minutes on a T4. If you run out of time, add "
       "`--model distilroberta-base` (about 3× faster, slightly lower accuracy)."),
    code("!python -m classifier.train --label queue --push-to-hub {HF_USER}/servicedesk-queue-classifier"),
    md("## 6. Fine-tune the priority classifier"),
    code("!python -m classifier.train --label priority --push-to-hub {HF_USER}/servicedesk-priority-classifier"),
    md("## 7. Results for the report"),
    code("!python -m classifier.compare --label queue\n!python -m classifier.compare --label priority"),
    code("from IPython.display import Markdown\n"
         "for lab in [\"queue\", \"priority\"]:\n"
         "    display(Markdown(open(f\"reports/metrics/comparison_{lab}.md\").read()))\n"
         "    display(Image(f\"reports/figures/confusion_matrix_{lab}.png\", width=650))\n"
         "    display(Image(f\"reports/figures/training_curves_{lab}.png\", width=650))"),
    code("import json\nfor lab in [\"queue\", \"priority\"]:\n"
         "    m = json.load(open(f\"reports/metrics/finetuned_{lab}.json\"))\n"
         "    print(f\"\\n== {lab}: most confused pairs ==\")\n"
         "    for c in m[\"top_confusions\"]:\n"
         "        print(f\"{c['true']} -> {c['predicted']}: {c['count']} tickets\")\n"
         "        for ex in c[\"examples\"][:2]:\n"
         "            print(\"   •\", ex[:160])"),
    md("## 8. MLflow runs (screenshot this table for the report)"),
    code("import mlflow\nmlflow.set_tracking_uri(\"sqlite:///mlflow.db\")\n"
         "runs = mlflow.search_runs(experiment_names=[\"servicedesk-copilot\"])\n"
         "cols = [c for c in runs.columns if c.startswith((\"params.label\", \"params.base_model\", \"metrics.test_\", "
         "\"metrics.eval_macro_f1\", \"metrics.train_minutes\"))]\n"
         "runs[[\"run_id\", \"start_time\"] + cols]"),
    md("## 9. Download everything\nThe models are on the Hub already. This zip holds the metrics, charts and MLflow "
       "database for the report, plus the past tickets for the RAG index."),
    code("!zip -qr part2_results.zip reports mlflow.db data/processed/kb_tickets.csv models/*/README.md models/*/metrics.json models/*/label_map.json\n"
         "from google.colab import files\nfiles.download(\"part2_results.zip\")"),
    md("Next, on your laptop, set in `.env`:\n```\nCLASSIFIER_MODEL_ID=<HF_USER>/servicedesk-queue-classifier\n"
       "PRIORITY_MODEL_ID=<HF_USER>/servicedesk-priority-classifier\n```\n"
       "unzip it into the repo folder, then rebuild the RAG index with the past tickets: "
       "`python -m rag.ingest --tickets data/processed/kb_tickets.csv --reset`."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": []},
                                                "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/colab_finetune.ipynb")
print("written")
