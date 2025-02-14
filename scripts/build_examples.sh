#!/bin/bash

unzip -o ../example_data.zip -d ../

mkdir -p ../examples/

mkdir -p ../examples/example_process_multifasta/context
cp ../example_data/context/multifasta.fa ../examples/example_process_multifasta/context/multifasta.fa

mkdir -p ../examples/example_extract_features/context
cp ../example_data/context/multifasta.fa ../examples/example_extract_features/context/multifasta.fa
cp ../example_data/context/multifasta.fa ../examples/example_extract_features/context/train_multifasta.fa
cp ../example_data/context/multifasta.fa ../examples/example_extract_features/context/val_multifasta.fa
cp ../example_data/context/multifasta.fa ../examples/example_extract_features/context/test_multifasta.fa

mkdir -p ../examples/example_train/context
cp ../example_data/context/multifasta.fa ../examples/example_train/context/multifasta.fa
cp ../example_data/context/train_multifasta.fa ../examples/example_train/context/train_multifasta.fa
cp ../example_data/context/val_multifasta.fa ../examples/example_train/context/val_multifasta.fa
cp ../example_data/context/test_multifasta.fa ../examples/example_train/context/test_multifasta.fa
cp ../example_data/context/*.csv ../examples/example_train/context/

mkdir -p ../examples/example_authenticate/context
cp ../example_data/context/multifasta.fa ../examples/example_authenticate/context/multifasta.fa
cp ../example_data/context/train_multifasta.fa ../examples/example_authenticate/context/train_multifasta.fa
cp ../example_data/context/val_multifasta.fa ../examples/example_authenticate/context/val_multifasta.fa
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate/context/test_multifasta.fa
cp ../example_data/context/*.csv ../examples/example_authenticate/context/
mkdir -p ../examples/example_authenticate/auth/
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate/auth/auth_multifasta.fa

