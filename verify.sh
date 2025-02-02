#!/bin/sh -
script_dir="$(dirname $0)"
cd "${script_dir}"
nv-sglrun mvn verify
