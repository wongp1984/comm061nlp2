#! /bin/bash
tar -zcvf model.tar.gz tut4-model_cpu.pt
split -b 20M model.tar.gz "model.tar.gz.part"
