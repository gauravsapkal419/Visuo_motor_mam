"""
run_simulation
==============

Run one simulation of the mesoscopic visuomotor model from the command line.

    python run_simulation.py --g -14 --seed 4 --sim-time 700
    python run_simulation.py --network-spec my_network.json --sim-spec my_sim.json

Results are written to simulations/<label>/ (see simulation.py). If results
with the same parameters exist, they are reused unless --force is given.
"""
import argparse

from mesoscopic_model import MesoscopicModel


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network-spec', help='json file with network parameters')
    parser.add_argument('--sim-spec', help='json file with simulation parameters')
    parser.add_argument('--g', type=float, help='relative inhibitory weight')
    parser.add_argument('--cc-weights-factor', type=float)
    parser.add_argument('--seed', type=int, help='rng seed')
    parser.add_argument('--sim-time', type=float, help='simulation time (ms)')
    parser.add_argument('--threads', type=int, help='NEST threads')
    parser.add_argument('--force', action='store_true',
                        help='simulate again even if results exist')
    args = parser.parse_args()

    network_spec = args.network_spec
    sim_spec = args.sim_spec
    if network_spec is None:
        conn = {}
        if args.g is not None:
            conn['g'] = args.g
        if args.cc_weights_factor is not None:
            conn['cc_weights_factor'] = args.cc_weights_factor
        network_spec = {'connection_params': conn} if conn else None
    if sim_spec is None:
        sim_spec = {}
        if args.seed is not None:
            sim_spec['rng_seed'] = args.seed
        if args.sim_time is not None:
            sim_spec['sim_time_ms'] = args.sim_time
        if args.threads is not None:
            sim_spec['local_num_threads'] = args.threads
        sim_spec = sim_spec or None

    M = MesoscopicModel(network_spec=network_spec, simulation=True,
                        sim_spec=sim_spec)
    M.simulation.simulate(force=args.force)
    print(f"label: {M.simulation.label}")


if __name__ == '__main__':
    main()
