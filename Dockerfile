# Dockerfile para Caddy com módulo DNS DuckDNS
FROM caddy:latest

RUN caddy add-package github.com/caddy-dns/duckdns
