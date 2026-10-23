# Account Lockout

Applies to: corporate accounts locked after repeated failed sign-ins.

## Why accounts get locked
Your account locks for 30 minutes after 5 failed sign-in attempts within 15 minutes. The most common cause is an old password saved on a phone, tablet, or mapped drive that keeps retrying in the background.

## Unlocking your account
1. Wait 30 minutes; the lock clears automatically.
2. Or unlock it immediately at `https://passwordreset.acme-corp.com` by choosing "Unlock my account" and verifying with MFA.

## Stopping repeat lockouts
- Remove and re-add your corporate email account on your phone.
- Forget and reconnect the "ACME-Corp" Wi-Fi network on all devices.
- On Windows, open Credential Manager and delete saved entries that start with `acme-corp`.

## When to contact IT
If your account locks again within an hour of unlocking it, more than three times in one day, contact the Service Desk so an engineer can trace the device sending bad passwords.
