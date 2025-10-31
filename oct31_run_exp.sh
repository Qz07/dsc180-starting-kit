#!/bin/bash
echo "Running RMU with different k values"
echo "-----------------------------------"
echo "k=0.25"
python -u run_exp.py --method rmu --k 0.25
echo "-----------------------------------"
echo "k=0.5"
python -u run_exp.py --method rmu --k 0.5
echo "-----------------------------------"
echo "k=0.75"
python -u run_exp.py --method rmu --k 0.75
echo "-----------------------------------"
echo "k=1"
python -u run_exp.py --method rmu --k 1
echo "-----------------------------------"
echo "Original unlearning method"
python -u run_exp.py --method original
echo "-----------------------------------"
echo "Retrain method"
python -u run_exp.py --method retrain
echo "-----------------------------------"

