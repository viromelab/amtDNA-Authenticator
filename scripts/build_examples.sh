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

mkdir -p ../examples/example_authenticate_multifasta/context
cp ../example_data/context/multifasta.fa ../examples/example_authenticate_multifasta/context/multifasta.fa
cp ../example_data/context/train_multifasta.fa ../examples/example_authenticate_multifasta/context/train_multifasta.fa
cp ../example_data/context/val_multifasta.fa ../examples/example_authenticate_multifasta/context/val_multifasta.fa
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate_multifasta/context/test_multifasta.fa
cp ../example_data/context/*.csv ../examples/example_authenticate_multifasta/context/
mkdir -p ../examples/example_authenticate_multifasta/auth/
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate_multifasta/auth/multifasta.fa
mkdir -p ../examples/example_authenticate_multifasta/context/models/
cp ../example_data/context/models/* ../examples/example_authenticate_multifasta/context/models/

mkdir -p ../examples/example_authenticate_single_fasta/context
cp ../example_data/context/multifasta.fa ../examples/example_authenticate_single_fasta/context/multifasta.fa
cp ../example_data/context/train_multifasta.fa ../examples/example_authenticate_single_fasta/context/train_multifasta.fa
cp ../example_data/context/val_multifasta.fa ../examples/example_authenticate_single_fasta/context/val_multifasta.fa
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate_single_fasta/context/test_multifasta.fa
cp ../example_data/context/*.csv ../examples/example_authenticate_single_fasta/context/
mkdir -p ../examples/example_authenticate_single_fasta/auth/
cp ../example_data/context/test_multifasta.fa ../examples/example_authenticate_single_fasta/auth/multifasta.fa
mkdir -p ../examples/example_authenticate_single_fasta/context/models/
cp ../example_data/context/models/* ../examples/example_authenticate_single_fasta/context/models/
cp ../example_data/I2473.fa ../examples/example_authenticate_single_fasta/