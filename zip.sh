#!/bin/bash
#SBATCH --job-name "zipping"
#SBATCH --cpus-per-task 16
#SBATCH --time 0-04:00:00
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

echo "Start zipping"
# J1, J2, J2_D_gam
zip -r v_vi.zip results/value_functions -i 'results/value_functions/v_*_vi.npz'
echo "Done with v_vi"
zip -r v_ospi.zip results/value_functions -i 'results/value_functions/v_*_ospi.npz'
echo "Done with v_ospi"
zip -r v_ospi.zip results/value_functions -i 'results/value_functions/v_*_ospi_cons.npz'
echo "Done with v_ospi_cons"
zip -r v_ospi.zip results/value_functions -i 'results/value_functions/v_*_ospi_lin.npz'
echo "Done with v_ospi_lin"
zip -r v_ospi.zip results/value_functions -i 'results/value_functions/v_*_ospi_abs.npz'
echo "Done with v_ospi_abs"
zip -r v_pi.zip results/value_functions -i 'results/value_functions/v_*_pi.npz'
echo "Done with v_pi"
zip -r v_fcfs.zip results/value_functions -i 'results/value_functions/v_*_fcfs.npz'
echo "Done with v_fcfs"

zip -r w_v.zip results/value_functions -i 'results/value_functions/w_*_vi.npz'
echo "Done with w_v"
zip -r w_ospi.zip results/value_functions -i 'results/value_functions/w_*_ospi.npz'
echo "Done with w_ospi"
zip -r w_pi.zip results/value_functions -i 'results/value_functions/w_*_pi.npz'
echo "Done with w_pi"
zip -r w_fcfs.zip results/value_functions -i 'results/value_functions/w_*_fcfs.npz'
echo "Done with w_fcfs"

zip -r pi_v.zip results/value_functions -i 'results/value_functions/pi_*_vi.npz'
echo "Done with pi_v"
zip -r pi_ospi.zip results/value_functions -i 'results/value_functions/pi_*_ospi.npz'
echo "Done with pi_ospi"
zip -r pi_ospi.zip results/value_functions -i 'results/value_functions/pi_*_ospi_cons.npz'
echo "Done with pi_ospi_cons"
zip -r pi_ospi.zip results/value_functions -i 'results/value_functions/pi_*_ospi_lin.npz'
echo "Done with pi_ospi_lin"
zip -r pi_ospi.zip results/value_functions -i 'results/value_functions/pi_*_ospi_abs.npz'
echo "Done with pi_ospi_abs"
zip -r pi_pi.zip results/value_functions -i 'results/value_functions/pi_*_pi.npz'
echo "Done with pi_pi"

# All g
#zip -r g.zip results/value_functions -i 'results/value_functions/g_*.npz'
#echo "Done with g"