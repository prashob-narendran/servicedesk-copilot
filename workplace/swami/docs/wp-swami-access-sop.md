---
title: Workplace SOP – Learning Platform Access and SSO Issues (illustrative)
doc_id: wp-swami-access-sop
---
# Workplace SOP – Learning Platform Access and SSO Issues

Illustrative procedure written for the Situated Learning section; it contains no confidential content.

## Learner sees "not authorised" on an enrolled course

1. Check the learner's enrolment status in the admin console: enrolment requests that need manager approval stay "pending" until approved.
2. Check that the enrolment is for the correct course version; retired versions are not visible to learners.
3. Check the learner's role: course content needs the Learner role in the client's organisation unit.
4. Ask the learner to sign out, clear the browser cache and sign in again; role changes apply at the next sign-in.
5. If all checks pass, escalate to Level 2 with the learner ID and course ID.

## SSO login failures after a client changes identity provider

A 403 error after the SSO page usually means the platform does not recognise the new provider's certificate or user identifier. Ask the client IT admin for the new metadata file, update the SSO configuration for that client, and test with one user before telling the client the fix is live. Treat an SSO failure that blocks all users of a client as a P1 incident.

## Account security

Never share passwords or one-time links in a case comment. Password resets are sent by the platform directly to the learner's registered email address.
