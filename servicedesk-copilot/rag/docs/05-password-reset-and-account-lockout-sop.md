---
title: SOP – Password Reset and Account Lockout
doc_id: sop-password
---
# SOP – Password Reset and Account Lockout

## Self-service first

Users should reset their own password through the self-service password portal, which requires their registered MFA method. Agents should guide users to self-service before performing a manual reset.

## Manual reset by an agent

1. Verify identity: the user must answer two of the three registered security questions, or the request must come from their line manager's verified email address. Never reset a password based on a phone call alone.
2. Reset the password in the identity console and tick "User must change password at next sign-in".
3. Send the temporary password by SMS to the registered mobile number only. Never send passwords by email or chat.
4. Record the verification method used in the ticket.

## Account lockout

Accounts lock for 30 minutes after 5 failed sign-in attempts. Agents can unlock an account early after identity verification. If an account locks repeatedly (3 times in 24 hours), check for an old password saved on a phone or another device, and inform Security if sign-in attempts come from unknown locations.

## Password policy

Passwords must be at least 12 characters, cannot reuse the last 10 passwords, and expire every 90 days for privileged accounts. Standard accounts do not expire if MFA is enabled.
