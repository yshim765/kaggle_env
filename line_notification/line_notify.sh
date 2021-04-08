#!/bin/sh
cd `dirname $0`

. ./setting.conf

if [ ! -e ./result ]; then
  echo "This is your first execution."
  kaggle kernels list --competition $COMPETITION_NAME --sort-by 'dateCreated' > ./result
  echo "Now, first result can be gained."
else
  mv ./result ./pastResult
  kaggle kernels list --competition $COMPETITION_NAME --sort-by 'dateCreated' > ./result
  python ./post_notify_to_line.py $LINE_NOTIFICATION_TOKEN
fi
