#!/bin/bash
# Check out and patch workload sources.
#
#   scripts/setup.sh <workload>... | all
#
# Each workload is cloned from UPSTREAM_URL at UPSTREAM_COMMIT (from
# workloads/<workload>/workload.sh) into $WORKLOAD_SRC_DIR/<workload>, the
# patches in workloads/<workload>/patches/ are applied, and its wl_setup hook
# runs inside the checkout.

source "$(dirname "$0")/lib.sh"

[[ $# -gt 0 ]] || die "usage: $0 <workload>... | all"
if [[ $1 == all ]]; then
    set -- $(for d in "$MEMPRINT_ROOT"/workloads/*/workload.sh; do basename "$(dirname "$d")"; done | grep -v TEMPLATE)
fi

for name in "$@"; do
    (
        load_workload "$name"
        if [[ -d $SRC ]]; then
            log "$name: $SRC already exists, skipping checkout"
        else
            log "$name: cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT"
            git clone -q "$UPSTREAM_URL" "$SRC" || die "clone failed"
            git -C "$SRC" checkout -q "$UPSTREAM_COMMIT" || die "checkout failed"
            shopt -s nullglob
            for patch in "$WORKLOAD_DIR"/patches/*.patch; do
                log "$name: applying $(basename "$patch")"
                git -C "$SRC" apply "$patch" || die "patch $(basename "$patch") failed"
            done
        fi
        cd "$SRC" && wl_setup || die "$name: setup hook failed"
        log "$name: ready in $SRC"
    ) || exit 1
done
