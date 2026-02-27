
#!/bin/bash
echo "density,gflops" > plot_data.csv

# Run for different densities
for d in 0.01 0.02 0.04 0.06 0.08 0.10; do
    echo -n "$d," >> plot_data.csv
    ./spmm_opt $d | grep "Throughput" | awk '{print $1}' >> plot_data.csv
done