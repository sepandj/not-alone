#!/bin/sh
# Fetch the free DB-IP "IP to City Lite" database (CC BY 4.0, no account
# needed). DB-IP publishes monthly; early in a month the new file may not be
# out yet, so fall back to last month's. Swaps in atomically, then restarts
# the app so it picks up the new file.
set -eu
DEST=/opt/not-alone/dbip-city-lite.mmdb
TMP=$(mktemp)
trap 'rm -f "$TMP" "$TMP.mmdb"' EXIT
for M in "$(date +%Y-%m)" "$(date -d 'last month' +%Y-%m)"; do
  if curl -sfL -o "$TMP" "https://download.db-ip.com/free/dbip-city-lite-$M.mmdb.gz"; then
    gunzip -c "$TMP" > "$TMP.mmdb"
    if [ -f "$DEST" ] && cmp -s "$TMP.mmdb" "$DEST"; then exit 0; fi
    install -m 0644 "$TMP.mmdb" "$DEST.new" && mv -f "$DEST.new" "$DEST"
    systemctl try-restart not-alone || true
    exit 0
  fi
done
echo "geoip-update: no DB-IP release found" >&2
exit 1
