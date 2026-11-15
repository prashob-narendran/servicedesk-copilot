---
title: Workplace Runbook – ETL Failures, Data Discrepancies and Data Residency (illustrative)
doc_id: wp-prashob-etl-runbook
---
# Workplace Runbook – ETL Failures, Data Discrepancies and Data Residency

Illustrative runbook written for the Situated Learning section; it contains no confidential content.

## First triage steps for a failed ETL pipeline

1. Check the orchestration tool's run log for the failed task and the exact error message.
2. Check Snowflake query history for the failed statement; warehouse timeouts usually mean the warehouse size or the query needs tuning.
3. Confirm whether upstream source files or tables arrived on time.
4. Re-run the failed task only if the source data is complete; otherwise wait for the source and inform the business owner.
5. Identify downstream reports and dashboards that depend on the pipeline, and post a data-delay notice for them.

Failures that block reports used in daily operations reviews are treated as high priority and must be acknowledged within 30 minutes.

## Investigating a data discrepancy

Compare the figure at each layer: source system, staging, warehouse model and report. The first layer where the figures differ shows where the problem started. Record the root cause and the fix in the ServiceNow ticket so similar tickets can be resolved faster.

## Data residency

Operational and customer data must remain within the organisation's approved environments located in the UAE. Copying data to any cloud environment or AI service hosted outside the UAE is not permitted, including for one-off analysis. AI tools may be used only when they are hosted in the UAE and approved by information security.
