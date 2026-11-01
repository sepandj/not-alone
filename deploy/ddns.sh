#!/usr/bin/env bash
# Dynamic DNS: keep notalone.sepand.dev pointed at this network's current public
# IP. Comcast residential IPs change occasionally; this closes that gap.
#
# Reads the GoDaddy Personal Access Token from /opt/not-alone/.godaddy:
#   GD_SECRET=gd_pat_xxx
# That file is root-only and is NOT stored in git.
set -euo pipefail

DOMAIN="sepand.dev"
NAME="notalone"
CREDS="/opt/not-alone/.godaddy"
TTL=600

# shellcheck disable=SC1090
source "$CREDS"

# Current public IP (try a couple of providers for resilience).
ip="$(curl -s4 --max-time 10 https://api.ipify.org || true)"
[ -z "$ip" ] && ip="$(curl -s4 --max-time 10 https://ifconfig.me || true)"
if ! [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
	echo "could not determine public IP" >&2
	exit 1
fi

current="$(curl -s --max-time 10 -H "Authorization: Bearer ${GD_SECRET}" \
	"https://api.godaddy.com/v1/domains/${DOMAIN}/records/A/${NAME}" \
	| grep -o '"data":"[0-9.]*"' | head -1 | cut -d'"' -f4 || true)"

if [ "$ip" = "$current" ]; then
	echo "$(date -Is) unchanged: $NAME.$DOMAIN -> $ip"
	exit 0
fi

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X PUT \
	-H "Authorization: Bearer ${GD_SECRET}" -H "Content-Type: application/json" \
	"https://api.godaddy.com/v1/domains/${DOMAIN}/records/A/${NAME}" \
	-d "[{\"data\":\"${ip}\",\"ttl\":${TTL}}]")"

if [ "$code" = "200" ]; then
	echo "$(date -Is) updated: $NAME.$DOMAIN $current -> $ip"
else
	echo "$(date -Is) update FAILED (HTTP $code)" >&2
	exit 1
fi
