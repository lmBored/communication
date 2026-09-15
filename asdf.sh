#!/bin/bash
set -euo pipefail

# Sync this repo to Snellius and submit a short escape-room training / SPS job.
REMOTE_ROOT="/home/knguyen2/madrona_escape_room"
REMOTE_HOST="snellius.surf.nl"

rsync -azP \
  --exclude '.git' \
  --exclude 'bin' \
  --exclude 'logs' \
  --exclude '.venv' \
  --exclude '.venv-*' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '*.pyc' \
  --exclude '*.a' \
  --exclude '.output.txt' \
  --exclude 'build' \
  --exclude 'ckpts' \
  --exclude 'checkpoints' \
  --exclude 'external' \
  --exclude 'CMakeCache.txt' \
  --exclude 'CMakeFiles' \
  --exclude 'cmake_install.cmake' \
  --exclude 'ImportExecutables.cmake' \
  --exclude 'Makefile' \
  --exclude 'link-make' \
  --exclude 'trained.mp4' \
  --exclude '.junie' \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"

SBATCH_EXPORT="ALL"
for name in NUM_ENVS NUM_UPDATES STEPS_PER_UPDATE; do
  if [ -n "${!name:-}" ]; then
    SBATCH_EXPORT+=",$name=${!name}"
  fi
done

JOB_ID=$(ssh "$REMOTE_HOST" "mkdir -p $REMOTE_ROOT/logs && cd $REMOTE_ROOT && sbatch --parsable --export='$SBATCH_EXPORT' hpc/comm.sbatch")

echo "job: $JOB_ID"
echo "workload: NUM_ENVS=${NUM_ENVS:-512} NUM_UPDATES=${NUM_UPDATES:-30} STEPS_PER_UPDATE=${STEPS_PER_UPDATE:-16}"
echo "ssh $REMOTE_HOST 'tail -f $REMOTE_ROOT/logs/escape_${JOB_ID}.err'"
if command -v pbcopy >/dev/null; then
  echo "ssh $REMOTE_HOST 'tail -f $REMOTE_ROOT/logs/escape_${JOB_ID}.err'" | pbcopy || true
fi
echo "ssh $REMOTE_HOST 'sstat -j ${JOB_ID}.batch --format=JobID,AveCPU,AveRSS,MaxRSS -P'"
echo "ssh $REMOTE_HOST 'squeue -j ${JOB_ID} -o \"%.18i %.9P %.24j %.8T %.10M %.6D %R\"'"
echo "ssh $REMOTE_HOST 'grep -E \"SPS|RESULT|SUMMARY|error|Error|Traceback\" $REMOTE_ROOT/logs/escape_${JOB_ID}.out $REMOTE_ROOT/logs/escape_${JOB_ID}.err | tail -50'"
