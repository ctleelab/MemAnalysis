#!/usr/bin/env bash
#
# MemAnalysis
#
# Copyright 2024- The MemAnalysis Authors
# and the project initiators Carolina Sarto and Christopher T. Lee.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Please help us support MemAnalysis development, by citing the research
# papers on the package. Check out https://github.com/ctleelab/MemAnalysis/.

# This script runs copyright header checks on modified files and
# reports/applies the necessary changes. It is shamelessly borrowed
# from the GROMACS project.
#
# See `copyright.sh -h` for a brief usage description.

# Parse command-line arguments
function usage() {
    echo "usage: copyright-precommit-check.sh {FILES}"
}

srcdir=`git rev-parse --show-toplevel`
pushd $srcdir >/dev/null
admin_dir=$srcdir/devtools

FILES=$(echo "$@" | tr " " "\n")
FILTERED_FILES=$(echo "$FILES" | git check-attr --stdin filter | sed -e 's/.*: filter: //'| paste <(echo "$FILES") - | grep -E 'copyright$' | cut -f1)

if [ -z "$FILTERED_FILES" ]
then
    popd >/dev/null
    exit 0
fi

if ! $admin_dir/copyright.py -F <(echo "$FILTERED_FILES") --add-missing --update-header
then
    echo "Copyright checking failed!"
    exit 2
fi

# Get back to the original directory
popd >/dev/null
exit 0
