# DIBLE Local Launch

> DIBLE is currently experimental research software. Do not use the research cryptography to protect real secrets.

## Start

```bash
python3 -m pip install -r requirements-server.txt
python3 -m dible_server.app
```

In another terminal:

```bash
python3 -m dible_server.cli bootstrap --organization DIBLE --email admin@example.test --password 'replace-with-a-strong-unique-password'
python3 -m dible_server.cli login --email admin@example.test --password 'replace-with-a-strong-unique-password'
```

Or start the container:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

## API

`GET /health`, `POST /v1/bootstrap`, `POST /v1/auth/login`, `GET/POST /v1/devices`, `POST /v1/devices/{id}/revoke`, `GET/POST /v1/vaults`, `POST /v1/artifacts`, and `GET /v1/audit`.
