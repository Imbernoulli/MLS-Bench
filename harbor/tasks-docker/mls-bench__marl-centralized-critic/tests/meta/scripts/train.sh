#!/bin/bash
set -e

case "${ENV}" in
  mmm)
    # Heterogeneous 10-agent map (1 Medivac + 2 Marauder + 7 Marine).
    # Tests coordination across diverse unit types.
    MAP="MMM"
    DEFAULT_TMAX=5050000
    ;;
  2s3z)
    # Medium heterogeneous symmetric. Yu et al. 2022: 5M steps.
    MAP="2s3z"
    DEFAULT_TMAX=5050000
    ;;
  3s5z)
    # Hard heterogeneous symmetric (team of 8). Yu et al. 2022: 10M steps;
    # reduced to 5M for practical wall-time with GPU packing (compute=0.4).
    MAP="3s5z"
    DEFAULT_TMAX=5050000
    ;;
  *)
    echo "Unknown ENV label: ${ENV}" >&2
    exit 1
    ;;
esac

# TMAX_OVERRIDE lets smoke runs / debug sessions shorten training
# without touching the script. Set via APPTAINERENV_TMAX_OVERRIDE on the host.
TMAX="${TMAX_OVERRIDE:-${DEFAULT_TMAX}}"
TEST_INTERVAL="${TEST_INTERVAL_OVERRIDE:-50000}"

python src/main.py --config=custom_mappo --env-config=smaclite \
    with env_args.map_name="${MAP}" \
    t_max=${TMAX} \
    test_interval=${TEST_INTERVAL} \
    test_nepisode=32 \
    seed=${SEED:-42} \
    common_reward=True \
    use_cuda=True
