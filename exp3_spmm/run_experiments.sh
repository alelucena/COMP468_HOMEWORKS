
#!/bin/bash
echo "density,gflops" > opt_data.csv
echo "density,gflops" > baseline_data.csv


# Run for different densities - opt and baseline
for d in 0.01 0.02 0.04 0.06 0.08 0.10; do
    echo -n "$d," >> opt_data.csv
    ./spmm_opt $d | grep "Throughput" | awk '{print $6}' >> opt_data.csv

    echo -n "$d," >> baseline_data.csv
    ./spmm_baseline $d | grep "Throughput" | awk '{print $6}' >> baseline_data.csv
done

