| Ticket | Should go to | Expected priority | Copilot queue (confidence) | Copilot priority | Sentiment | PII masked |
|---|---|---|---|---|---|---|
| RB-1 ATM did not dispense cash but account debited | Disputes - ATM and debit (EFT) | medium | Billing and Payments (78%) | low | neutral | PERSON, ACCOUNT_NUMBER, PHONE |
| RB-2 Unrecognised credit card charge | Card disputes and fraud | high | Billing and Payments (100%) | medium | negative | PERSON, EMAIL |
| RB-3 Zelle payment sent to a scammer | Payments disputes (Zelle / P2P) | high | Billing and Payments (100%) | medium | negative | PERSON, PERSON, PERSON, PERSON |
| RB-4 Mobile banking login failing for many customers | IT incident - digital banking | high | Product Support (36%) ⚠ review | high | negative | PERSON, PERSON, PERSON, PERSON |
| RB-5 Duplicate card charge at restaurant | Card disputes and fraud | low | Billing and Payments (100%) | medium | negative | PERSON |

## Summaries and draft replies

**RB-1** summary: ATM error debited $400 from [ACCOUNT_NUMBER_1] without dispensing cash, customer [PERSON_1] wants refund; urgent callback to [PHONE_1].

Reply grounded: True · sources: Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative); Past ticket: Charging Discrepancy

**RB-2** summary: [PERSON_1] reports an unauthorized $1,249.99 charge on credit card ending 4417, requests reversal and card blocking due to fraud concerns, emails [EMAIL_1].

Reply grounded: True · sources: Past ticket: Invoice Discrepancy Concern; Past ticket: Support ticket; Past ticket: Discrepancy in Invoice Billing; Workplace SOP – Card and Electronic Payment Disputes (illustrative)

**RB-3** summary: [PERSON_1] sent $850 via Zelle to a suspected scammer [PERSON_2] for a sofa, who vanished; requests urgent reversal after two upset calls.

Reply grounded: True · sources: Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative); TechServ Incident Management Process; Past ticket: Hospital System Security Measures

**RB-4** summary: Mobile banking login failing for many [PERSON_1]; since 08:30, [PERSON_2] reports a 25‑minute call queue and “service unavailable” errors—urgent IT investigation required.

Reply grounded: True · sources: Past ticket: Reported Login Failures During Peak Hours; Past ticket: Report of Login Problem; Past ticket: Report on Frequent Login Issues; SOP – VPN and Remote Access Troubleshooting

**RB-5** summary: Duplicate $86.40 charge on [PERSON_1]’s credit card ending 2290 for a restaurant bill on 9 November; she requests removal with receipt.

Reply grounded: True · sources: TechServ Billing, Refunds and Payments Policy; Workplace SOP – Card and Electronic Payment Disputes (illustrative); Past ticket: Charging Discrepancy; Past ticket: Invoice Discrepancy Concern


## Knowledge-base questions

**Q:** What is the deadline to give provisional credit for an ATM dispute?

**A:** Give the provisional credit if the investigation is not finished within 10 business days. [1]

Sources: Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace Runbook – Digital Banking Downtime (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative)

**Q:** What must the agent collect from a customer reporting an unauthorised card transaction?

**A:** The agent must collect the following information:

1. Verify the caller's identity using the bank's call‑centre verification steps.  
2. Record the transaction date, amount, merchant or ATM location, and the channel (ATM, debit card, credit card, Zelle).  
3. Ask whether the customer still has the card and whether the transaction was authorised by them.  
4. For suspected fraud, block the card immediately and order a replacement.  

[2]

Sources: Workplace SOP – Card and Electronic Payment Disputes (illustrative); Workplace SOP – Card and Electronic Payment Disputes (illustrative); Past ticket: Urgent: Reporting Unapproved Access Attempts Required; Past ticket: Unexplained Duplicate Charges

**Q:** Can a Zelle payment sent to a scammer be reversed?

**A:** The policy states that if the customer authorized the Zelle payment, it is treated like a normal electronic transfer and generally cannot be reversed or refunded on the call. Only unauthorised transfers are handled as disputes. [1]

Sources: Workplace SOP – Card and Electronic Payment Disputes (illustrative); Past ticket: Modification Inquiry; Past ticket: Charging Discrepancy; Past ticket: Hospital System Security Measures
