# Shared Drives and File Access

Applies to: SharePoint sites, OneDrive, and legacy network drives (S: drive).

## Requesting access to a shared folder
Access is approved by the folder owner, not IT.
1. Open the link to the folder or SharePoint site.
2. Click "Request access" and add a short reason.
3. The owner gets an email and can approve it. Access applies within 15 minutes.
If you do not know the owner, raise an "Access Request" on the IT portal with the folder path.

## Mapped network drive missing
- Connect to the VPN if you are remote; network drives only work on the corporate network.
- Open File Explorer, right-click "This PC", choose "Map network drive", and enter `\\fs01.acme-corp.local\shared` for the S: drive.

## OneDrive not syncing
1. Click the OneDrive cloud icon in the system tray and check for errors.
2. Pause and resume syncing.
3. Make sure file names do not contain characters like `" * : < > ? / \ |` and paths are shorter than 400 characters.
4. Sign out of OneDrive and sign back in.

## Recovering deleted files
Deleted files stay in the OneDrive or SharePoint Recycle Bin for 93 days. Restore them from the Recycle Bin on the website.
