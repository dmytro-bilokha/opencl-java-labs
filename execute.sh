#!/bin/sh -
script_dir="$(dirname $0)"
class_name="${1}"
shift
nv-sglrun java --enable-native-access=com.dmytrobilokha --module-path "${script_dir}/target/opencl.jar" --module com.dmytrobilokha/com.dmytrobilokha.${class_name} $@
