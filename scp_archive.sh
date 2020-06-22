#!/usr/bin/env bash

echo "scp-ing the zipped archive from IP $1"
scp williapb@$1:/home/williapb/phd/pycharm/pytorch/archive.zip ~/repos/phd/pycharm/old_archives/

