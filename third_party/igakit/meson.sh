#!/bin/sh
set -eu
test -f meson.build
rm -rf build install
options=(
    --python.platlibdir=""
    --python.bytecompile=-1
)
meson setup build --prefix="$PWD/install" "${options[@]}"
meson compile -C build
meson install -C build
