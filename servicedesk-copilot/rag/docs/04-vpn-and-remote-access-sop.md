---
title: SOP – VPN and Remote Access Troubleshooting
doc_id: sop-vpn
---
# SOP – VPN and Remote Access Troubleshooting

Applies to the TechServ-managed GlobalConnect VPN client used by customers' remote staff.

## First-line checks (IT Support)

1. Confirm the user has a working internet connection without the VPN (open any public website).
2. Ask for the exact error message and the VPN client version (Help > About).
3. Check whether the user's account is locked or the password expired; if so, follow the password-reset SOP first.
4. Ask the user to sign out of the VPN client, restart the laptop and connect again.
5. If the client version is older than the current approved version, reinstall it from the company software portal.

## Common issues

### VPN disconnects every few minutes
Usually caused by an unstable Wi-Fi connection or a power-saving setting on the network adapter. Ask the user to switch to a wired connection or another network, and to disable "Allow the computer to turn off this device to save power" for the Wi-Fi adapter. If several users report disconnections at the same time, treat it as a possible outage and inform Service Outages and Maintenance.

### Authentication failed
Check that multi-factor authentication (MFA) is registered and the authenticator app time is correct. Re-register MFA through the self-service portal if the user changed phones.

### Connected but cannot reach internal applications
Ask the user to run the client's built-in diagnostics and send the report. Escalate to Technical Support (Level 2 network team) with the report attached.

## Escalation

If first-line checks do not fix the issue within 30 minutes, escalate to Level 2. If more than 10 users from one customer are affected, raise a P1 and follow the Incident Management Process.
