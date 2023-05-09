#! /bin/bash
cat model.tar.gz.parta* > model.tar.gz.joined
tar -zxvf model.tar.gz.joined