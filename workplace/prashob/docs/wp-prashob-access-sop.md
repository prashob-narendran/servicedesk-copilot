---
title: Workplace SOP – Snowflake and Reporting Access Requests (illustrative)
doc_id: wp-prashob-access-sop
---
# Workplace SOP – Snowflake and Reporting Access Requests

Illustrative procedure written for the Situated Learning section; it reflects common practice in an enterprise data management team and contains no confidential content.

## Raising a request

All access requests are raised in ServiceNow using the "Data Platform Access" catalogue item. The request must name the platform (Snowflake, reporting tool or data integration tool), the database and schema or report, the access level (read, write or owner) and the business justification.

## Approvals

1. The requester's line manager approves the business need.
2. The data owner of the schema approves access to the data domain.
3. Requests for schemas containing personal data also need approval from the data governance office.

Read-only access to non-personal operational schemas is normally completed within 2 business days of the final approval. Write and owner access needs a change request and takes up to 5 business days.

## Granting access

Access is granted only through role-based access control: the user is added to an existing Snowflake functional role (for example ANALYST_PORT_OPS_READ). Direct grants to individual users are not allowed. Access is reviewed every quarter, and unused access is removed after 90 days.
