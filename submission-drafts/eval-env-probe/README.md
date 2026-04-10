# eval-env-probe

This is a deliberately failing minimal SOAR submission used to inspect the
evaluation machine environment from the last 50 log lines.

- `prepare_env.sh` does not modify the environment.
- `prepare_model.sh` prints targeted runtime/tool/package information and then
  exits with a non-zero status on purpose.
