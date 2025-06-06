#!/bin/bash
# 参考: dockerでvolumeをマウントしたときのファイルのowner問題（https://qiita.com/yohm/items/047b2e68d008ebb0f001）
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}
echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -g $GROUP_ID user
export HOME=/home/user
exec /usr/sbin/gosu user "$@"