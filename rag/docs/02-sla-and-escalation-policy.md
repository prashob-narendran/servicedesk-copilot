---
title: TechServ SLA and Escalation Policy
doc_id: sla-policy
---
# TechServ SLA and Escalation Policy

This policy defines ticket priorities, response and resolution targets, and when and how to escalate.

## Priority definitions

- **P1 – High (critical):** a production service is down or a security incident is in progress; many users, revenue or customer data are affected and there is no workaround.
- **P2 – High:** a major function is impaired for a team or a key customer; a workaround may exist but is not sustainable.
- **P3 – Medium:** a single user or a non-critical function is affected, or a time-bound request.
- **P4 – Low:** questions, how-to requests, feedback and routine service requests.

The copilot's priority labels map as follows: high = P1 or P2 (the agent decides which), medium = P3, low = P4.

## Response and resolution targets

| Priority | First response | Update frequency | Target resolution |
|---|---|---|---|
| P1 | 15 minutes, 24x7 | every 30 minutes | 4 hours |
| P2 | 1 hour, 24x7 | every 2 hours | 8 hours |
| P3 | 4 business hours | daily | 3 business days |
| P4 | 1 business day | on change | 10 business days |

Business hours are 08:00–18:00, Sunday to Thursday (UAE) and Monday to Friday (India), excluding public holidays. The SLA clock pauses while a ticket is in "Waiting for customer" status.

## Escalation

1. **Functional escalation:** if the assigned agent cannot progress a ticket within 50% of its resolution target, it moves to the Level 2 team of the same queue.
2. **Hierarchical escalation:** the shift lead is informed when 75% of the target has elapsed; the service delivery manager is informed when a P1 or P2 breaches its target.
3. **Major incident:** any P1 affecting more than one customer is declared a major incident and handled under the Incident Management Process.

Customers can request escalation at any time by replying "ESCALATE" to the ticket email or by calling the service desk; the request is logged and the shift lead responds within 1 hour.

## Service credits

If more than 5% of a customer's P1 and P2 tickets in a calendar month breach the resolution target, the customer is entitled to a service credit of 5% of that month's support fee.
