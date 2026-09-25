# Contributing to Crop Health Triage

Thank you for your interest in contributing to Crop Health Triage.

## Project Areas

- `backend/` — machine learning, inference API, model artifacts, and backend logic
- `mobile-app/` — Android application
- `web-app/` — web application

## Before Contributing

1. Read the repository README.
2. Check existing issues and pull requests.
3. Keep changes focused on one feature or fix.
4. Never commit secrets, credentials, datasets, build artifacts, or private configuration.
5. Test changes before opening a pull request.

## Development Workflow

```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Describe your change"
git push origin feature/your-feature-name
```

Open a pull request describing what changed, why it changed, how it was tested, and any limitations.

## Code Guidelines

### Python / Backend
- Keep functions focused and readable.
- Avoid hard-coded secrets.
- Document API behavior when endpoints change.
- Preserve model input/output formats unless intentionally changed.
- Test model-loading and inference changes.

### Android
- Keep API configuration separate from secrets.
- Do not commit `local.properties`.
- Test on a clean build where possible.
- Keep UI and networking changes modular.

### Web
- Do not commit environment secrets.
- Keep API configuration documented.
- Test major changes in a clean browser session.

## Machine Learning Contributions

For training/model changes, document the dataset/source, preprocessing, architecture, training configuration, evaluation results, and limitations.

Do not upload large raw datasets unless maintainers explicitly decide that redistribution is appropriate.

## Security

For security issues, follow [`SECURITY.md`](SECURITY.md) rather than publishing credentials or sensitive details in a public issue.

## License

By contributing, you agree that your contribution may be distributed under the project's MIT License.
