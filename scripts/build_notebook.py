"""Generates notebooks/colab_finetune.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Part II — Fine-tuning the ticket classifiers (Google Colab)\n"
       "AI Service Desk Copilot · Group 60 · AIMLZG549 Assignment II\n\n"
       "**Runtime → Change runtime type → T4 GPU** before you start. The whole notebook takes about 75–90 minutes.\n\n"
       "What it produces (all needed for the report):\n"
       "1. Clean train / validation / test splits and EDA charts\n"
       "2. Two baselines: TF-IDF + logistic regression, and a zero-shot LLM (free Groq API)\n"
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
       "- `GROQ_API_KEY` — free key from console.groq.com/keys, for the zero-shot baseline\n"
       "- `HF_TOKEN` — a Hugging Face token with **write** access\n"
       "- your Hugging Face username is read automatically from the token"),
    code("import os\nfrom google.colab import userdata\n"
         "os.environ[\"LLM_PROVIDER\"] = \"groq\"\n"
         "os.environ[\"GROQ_API_KEY\"] = userdata.get(\"GROQ_API_KEY\")\n"
         "os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")\n"
         "from huggingface_hub import login, whoami\nlogin(token=os.environ[\"HF_TOKEN\"])\n"
         "HF_USER = whoami()[\"name\"]  # your Hugging Face username, read from the token\n"
         "print(\"Models will be pushed to\", HF_USER)"),
    md("## 3. Prepare the data\nEnglish tickets only, empty and duplicate tickets removed, 80/10/10 split stratified by queue "
       "(seed 42). If loading fails, the error lists the CSV files in the dataset repo; re-run with `--hub-file <name>`."),
    code("!python -m data.prepare"),
    code("from IPython.display import Image, display\n"
         "for f in [\"queue_distribution\", \"priority_distribution\", \"ticket_length\"]:\n"
         "    display(Image(f\"reports/figures/{f}.png\", width=650))"),
    md("## 4. TF-IDF baselines (the 'before' numbers, about 2 minutes)"),
    code("!python -m classifier.baselines --label queue\n!python -m classifier.baselines --label priority"),
    md("## 5. Fine-tune the queue classifier\nAbout 30 minutes on a T4 with the final settings (learning rate 5e-5, "
       "8 epochs, batch 32, 128 tokens). A first run with 2e-5 / 4 epochs under-fitted; pass `--lr 2e-5 --epochs 4 "
       "--batch-size 16 --max-length 256` to reproduce it."),
    code("!python -m classifier.train --label queue --push-to-hub {HF_USER}/servicedesk-queue-classifier"),
    md("## 6. Fine-tune the priority classifier"),
    code("!python -m classifier.train --label priority --push-to-hub {HF_USER}/servicedesk-priority-classifier"),
    md("## 6b. Zero-shot LLM baseline (about 15–25 minutes)\nOptional. Scores the free Groq gpt-oss-20b model on the "
       "fixed 300-ticket sample. It is slow because of free-tier rate limits; the client "
       "waits and retries automatically. The TF-IDF run is repeated in seconds so both baselines land in one file set."),
    code("!python -m classifier.baselines --label queue --zero-shot"),
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
         "r = mlflow.search_runs(experiment_names=[\"servicedesk-copilot\"],\n"
         "                       filter_string=\"attributes.status = 'FINISHED'\")\n"
         "t = r[[\"params.label\", \"params.learning_rate\", \"params.num_train_epochs\",\n"
         "       \"params.per_device_train_batch_size\", \"params.max_length\",\n"
         "       \"metrics.eval_macro_f1\", \"metrics.test_macro_f1\", \"metrics.test_accuracy\",\n"
         "       \"metrics.train_minutes\", \"start_time\"]].copy()\n"
         "t.columns = [\"label\", \"learning rate\", \"epochs\", \"batch size\", \"max tokens\",\n"
         "             \"val macro-F1\", \"test macro-F1\", \"test accuracy\", \"train minutes\", \"started (UTC)\"]\n"
         "t[\"started (UTC)\"] = t[\"started (UTC)\"].dt.strftime(\"%Y-%m-%d %H:%M\")\n"
         "t.sort_values([\"label\", \"started (UTC)\"]).reset_index(drop=True).round(3)"),
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
