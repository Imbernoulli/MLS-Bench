#!/bin/bash
# Train BC-GMM with custom observation encoder using robomimic's native pipeline.
# Selects dataset + obs keys + rollout horizon based on ENV label.

case "${ENV}" in
  tool_hang_ph)
    DATASET="${DATASET_ROOT}/tool_hang/ph/low_dim_v15.hdf5"
    OBS_KEYS='["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]'
    HORIZON=700
    ;;
  can_ph)
    DATASET="${DATASET_ROOT}/can/ph/low_dim_v15.hdf5"
    OBS_KEYS='["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]'
    HORIZON=400
    ;;
  square_ph)
    DATASET="${DATASET_ROOT}/square/ph/low_dim_v15.hdf5"
    OBS_KEYS='["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]'
    HORIZON=400
    ;;
  *)
    DATASET="${DATASET_ROOT}/tool_hang/ph/low_dim_v15.hdf5"
    OBS_KEYS='["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]'
    HORIZON=700
    ;;
esac

RUN_LABEL="${ENV:-tool_hang_ph}"
OUTDIR="${OUTPUT_DIR:-/tmp/bc_output}/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_s${SEED:-42}"
CONFIG_PATH="${TMPDIR:-/tmp}/train_config_${RUN_LABEL}_s${SEED:-42}.json"
python -c "
import json
config = json.load(open('bc_gmm_config.json'))
config['train']['data'] = '${DATASET}'
config['train']['seed'] = ${SEED:-42}
config['train']['output_dir'] = '${OUTDIR}'
config['observation']['modalities']['obs']['low_dim'] = ${OBS_KEYS}
config['experiment']['rollout']['horizon'] = ${HORIZON}
json.dump(config, open('${CONFIG_PATH}', 'w'), indent=2)
"

python -m robomimic.scripts.train --config "${CONFIG_PATH}"
