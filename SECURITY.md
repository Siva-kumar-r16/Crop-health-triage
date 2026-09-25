# Security Policy

## Supported Versions

Crop Health Triage is a hackathon/research prototype. Security fixes will be considered for the current public version.

## Reporting a Security Issue

Do not publicly disclose credentials, tokens, private keys, or other sensitive information in a GitHub issue.

Contact the project maintainer privately with:
- a short description
- affected component or file
- reproduction steps, when safe
- potential impact
- suggested mitigation, if known

## Never Commit Secrets

This repository is public. Never commit:
- API keys
- access tokens
- passwords
- database credentials
- cloud credentials
- private certificates or keys
- Android signing keys
- secret-containing `.env` files
- `local.properties`
- deployment credentials
- authentication cookies or session data

Use environment variables or local configuration excluded by `.gitignore`.

## If a Secret Is Accidentally Published

Deleting a file is not enough if the credential was exposed.

1. Revoke or rotate the credential immediately.
2. Remove the secret from the working tree.
3. Review and clean Git history if necessary.
4. Check whether the credential was reused elsewhere.
5. Report the incident privately to maintainers.

## Model and Dataset Safety

Do not upload private datasets, personally identifiable information, confidential farm records, or proprietary data without authorization.

## Responsible Use

This is a crop-disease triage prototype. Predictions may be incorrect, especially on images that differ from training data. Users should not rely solely on an automated prediction for high-impact agricultural decisions.
