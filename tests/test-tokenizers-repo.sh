#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x "$toktest" ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

if [ -d "$folder" ] && [ -d "$folder/.git" ]; then
    (cd "$folder"; git pull) || exit 1
else
    git clone "$repo" "$folder" || exit 1

    # byteswap models if on big endian
    if [ "$(uname -m)" = s390x ]; then
        for f in $folder/*/*.gguf; do
            echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
        done
    fi
fi

status=0
tested=0
while IFS= read -r -d '' gguf; do
    if [ -f "$gguf.inp" ] && [ -f "$gguf.out" ]; then
        tested=$((tested + 1))
        "$toktest" "$gguf" || status=1
    else
        printf 'Found "%s" without matching inp/out files, ignoring...\n' "$gguf"
    fi
done < <(find "$folder" -type f -name '*.gguf' -print0)

if [ "$tested" -eq 0 ]; then
    printf 'No tokenizer fixtures with matching inp/out files found.\n' >&2
    exit 1
fi
exit "$status"

