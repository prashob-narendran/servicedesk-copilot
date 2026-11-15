| Ticket | Should go to | Expected priority | Copilot queue (confidence) | Copilot priority | Sentiment | PII masked |
|---|---|---|---|---|---|---|
| WP-1 Read access to terminal KPI schema in Snowflake | Data access (Snowflake platform) | low | Customer Service (39%) ⚠ review | medium | neutral | PERSON, EMAIL, PERSON |
| WP-2 Country-wise throughput dashboard shows wrong TEU figures | Reporting and BI | high | Product Support (48%) ⚠ review | medium | negative | PHONE |
| WP-3 Nightly ETL job VESSEL_CALLS_LOAD failed | ETL and data integration | high | Technical Support (41%) ⚠ review | medium | negative | — |
| WP-4 Duplicate container records after source migration | Data quality | medium | Returns and Exchanges (71%) | medium | negative | — |
| WP-5 Q3 export volumes by country for board pack | Business data request (Reporting and BI) | low | Customer Service (38%) ⚠ review | low | neutral | EMAIL |

## Summaries and draft replies

**WP-1** summary: Request: grant read-only access to PORT_OPS_DW.TERMINAL_KPI schema in Snowflake for analyst [PERSON_1] ([EMAIL_1]); manager [PERSON_2] approved, needed for weekly terminal productivity report.

Reply grounded: True · sources: Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative)

**WP-2** summary: Country-wise throughput dashboard shows 18% lower TEU for Terminal 2 in October; CEO office needs urgent investigation for tomorrow’s review, please call [PHONE_1].

Reply grounded: False · sources: 

**WP-3** summary: Nightly ETL job VESSEL_CALLS_LOAD failed due to Snowflake warehouse timeout, affecting berth utilisation and vessel turnaround reports; urgent restart and root‑cause investigation required.

Reply grounded: True · sources: Workplace Runbook – ETL Failures, Data Discrepancies and Data Residency (illustrative); Past ticket: Service Interruption; Past ticket: Several Tools Failed to Function Properly This Morning; Past ticket: Report of SaaS Platform Disruption

**WP-4** summary: Duplicate container records (~3,200) appear in IMPORT_MANIFEST post‑migration, causing customs reconciliation report mismatches; data quality team urgently needs to review deduplication rules.

Reply grounded: True · sources: Past ticket: Issues with Recent Inconsistent Data Analysis Reports; Past ticket: Campaign Metrics Problem; Past ticket: Problem Reporting; Past ticket: Problem with Unreliable Digital Engagement Statistics

**WP-5** summary: Prepare Q3 monthly export container volumes by destination country, split by terminal, for board pack due next week; contact [EMAIL_1] if unclear.

Reply grounded: True · sources: Past ticket: Inquiry on Digital Strategies for Brand Growth with Pro Tools and Docker; TechServ Incident Management Process; TechServ SLA and Escalation Policy; Past ticket: Include Voice Text Integration


## Knowledge-base questions

**Q:** How do I request read access to a Snowflake schema?

**A:** 1. Submit a request in ServiceNow using the **"Data Platform Access"** catalogue item.  
2. In the request form, specify:
   - **Platform**: Snowflake  
   - **Database** and **Schema** you need access to  
   - **Access level**: read  
   - **Business justification**  
3. Await approvals: line manager → data owner → (if personal data) data governance office [4].  
4. Once all approvals are obtained, read‑only access will be granted within ~2 business days [1].

Sources: Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative); Workplace SOP – Snowflake and Reporting Access Requests (illustrative)

**Q:** The nightly ETL job failed. What are the first triage steps?

**A:** 1. Check the orchestration tool’s run log for the failed task and the exact error message.  
2. Check Snowflake query history for the failed statement; warehouse timeouts usually mean the warehouse size or the query needs tuning.  
3. Confirm whether upstream source files or tables arrived on time.  
4. Re‑run the failed task only if the source data is complete; otherwise wait for the source and inform the business owner.  
5. Identify downstream reports and dashboards that depend on the pipeline, and post a data‑delay notice for them.【1】

Sources: Workplace Runbook – ETL Failures, Data Discrepancies and Data Residency (illustrative); Past ticket: Several Tools Failed to Function Properly This Morning; Past ticket: Service Interruption; Past ticket: Support ticket

**Q:** Can container data be copied to a cloud environment outside the UAE for analysis?

**A:** No, copying container data to a cloud environment outside the UAE is not permitted. [1]

Sources: Workplace Runbook – ETL Failures, Data Discrepancies and Data Residency (illustrative); Past ticket: Integration of Oracle Database 19c Docker for Enhancing Investment Analytics; Past ticket: Concerns About Data Analytics Application Performance; Past ticket: Enhancing Investments with Google Cloud
