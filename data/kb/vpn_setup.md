# VPN Setup and Troubleshooting

Applies to: all employees working remotely. Client: GlobalProtect.

## Installing the VPN client
1. Open the Company Portal (Windows) or Self Service (macOS).
2. Search for "GlobalProtect" and click Install.
3. When asked for the portal address, enter `vpn.acme-corp.com`.
4. Sign in with your corporate email and password, then approve the MFA prompt.

## VPN will not connect
- Check that you have a working internet connection by opening any public website.
- Make sure your password has not expired. An expired password causes "Authentication failed" errors.
- Disconnect, quit GlobalProtect from the system tray, and reopen it.
- If you see "Gateway unreachable", switch from hotel or guest Wi-Fi to a mobile hotspot; some public networks block VPN traffic.
- Restart your laptop if the client is stuck on "Connecting...".

## VPN connects but internal sites do not load
- Run `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (macOS).
- Make sure no personal VPN (NordVPN, etc.) is running at the same time.

## VPN disconnects frequently
Idle sessions are disconnected after 12 hours by policy. Frequent drops under 1 hour usually point to an unstable Wi-Fi connection; try a wired connection or move closer to the router.

## When to contact IT
Contact the Service Desk if the client shows "Certificate error" or "Your device is not compliant", because these require a device check by an engineer.
