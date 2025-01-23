#!/bin/sh -
script_dir="$(dirname $0)"
nv-sglrun java --enable-native-access=com.dmytrobilokha --module-path "${script_dir}/target/opencl.jar" --module com.dmytrobilokha/com.dmytrobilokha.HelloWorld
