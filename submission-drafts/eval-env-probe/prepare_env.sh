#!/usr/bin/env bash

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] cwd=$(pwd)"
echo "[prepare_env] python=$(command -v python || true)"
echo "[prepare_env] uv=$(command -v uv || true)"
echo "[prepare_env] done $(date '+%F %T')"
