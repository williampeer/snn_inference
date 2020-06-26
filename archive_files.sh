#!/usr/bin/env bash

mkdir archive
cp -r figures archive/
cp -r Logs archive/
cp -r saved archive/

rm -r figures/*/*
rm -r figures/*/
rm figures/*.png
rm Logs/*
rm -r saved/*/*
rm -r saved/*
mkdir saved/plot_data

zip -r archive.zip archive/
