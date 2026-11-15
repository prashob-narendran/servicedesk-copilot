| Ticket | Should go to | Expected priority | Copilot queue (confidence) | Copilot priority | Sentiment | PII masked |
|---|---|---|---|---|---|---|
| SW-1 Learner cannot open enrolled course | Access and accounts | medium | Sales and Pre-Sales (96%) | medium | neutral | PERSON, PERSON, EMAIL |
| SW-2 Learning platform down at dealer training centre | Platform operations (incident) | high | Returns and Exchanges (94%) | high | negative | PERSON |
| SW-3 Course progress not saved | Product and usability | medium | Product Support (33%) ⚠ review | medium | negative | PERSON, PERSON |
| SW-4 Video lessons stuck loading in Chrome | Technical support | medium | IT Support (66%) | medium | neutral | — |
| SW-5 SSO users cannot log in after migration | Access and SSO integration | high | Technical Support (47%) ⚠ review | high | negative | PERSON, PHONE |

## Summaries and draft replies

**SW-1** summary: [PERSON_1] cannot access enrolled 'EV Battery Safety – Level 2' course, receiving “not authorised” error; manager approved enrolment and assessment is due Friday.

Reply grounded: False · sources: 

**SW-2** summary: Learning platform down at dealer training centre, error 503 for 120 [PERSON_1] in scheduled certification session since 09:10—urgent fix required.

Reply grounded: True · sources: Workplace Runbook – Platform Incidents and Client SLAs (illustrative); Past ticket: Assistance Required for SaaS Platform; Past ticket: Assistance Required for Analytics Platform Malfunction; Past ticket: Login Problem with SaaS Platform

**SW-3** summary: Course progress not saved: [PERSON_1] loses module progress upon closing and reopening the [PERSON_2], requiring a restart; urgent request to enable resume functionality.

Reply grounded: False · sources: 

**SW-4** summary: Video lessons in the brake systems course are stuck at 0% loading in Chrome for the user, but play fine in Edge; urgent help needed.

Reply grounded: True · sources: Workplace Runbook – Platform Incidents and Client SLAs (illustrative); Workplace SOP – Learning Platform Access and SSO Issues (illustrative); Past ticket: Investment Dashboard Loading Issue; Past ticket: Sync Concern

**SW-5** summary: SSO users cannot log in to the learning platform post-migration, receiving a 403 error; 300 users affected, urgent help requested by IT admin [PERSON_1] ([PHONE_1]).

Reply grounded: True · sources: Workplace SOP – Learning Platform Access and SSO Issues (illustrative); Past ticket: Login Problem with SaaS Platform; Past ticket: Problem with Login on SaaS Platform; Past ticket: Problem with Login SaaS Platform


## Knowledge-base questions

**Q:** A learner sees 'not authorised' on a course they enrolled in. What should I check?

**A:** The knowledge base doesn't cover this question. Please escalate to a senior agent or check with the relevant team.

Sources: 

**Q:** What do we do when the learning platform is down for a client?

**A:** When the learning platform is down for a [PERSON_1]:

1. Check the cloud provider's health dashboard and our monitoring alerts. [1]  
2. Raise a P1 case in Salesforce and link all related client cases to it. [1]  
3. Post a notice on the platform status page within 30 minutes and email the client’s main contact. [1]  
4. If a live training or certification session is affected, offer to reschedule it without charge. [1]  
5. Update the client every hour until service is restored, and send a short incident report within 3 business days. [1]

Sources: Workplace Runbook – Platform Incidents and Client SLAs (illustrative); Past ticket: Assistance Required for SaaS Platform; Past ticket: Support ticket; Past ticket: Digital Platform Outage Problem

**Q:** How quickly must we respond to a P1 case for an enterprise client?

**A:** Respond within 30 minutes. [1]

Sources: Workplace Runbook – Platform Incidents and Client SLAs (illustrative); SOP – VPN and Remote Access Troubleshooting; TechServ Incident Management Process; Workplace Runbook – Digital Banking Downtime (illustrative)
