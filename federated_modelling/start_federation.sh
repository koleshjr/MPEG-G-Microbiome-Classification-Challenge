#!/bin/bash
# start_federation.sh
# -------------------
# Starts Flower XGBoost federation with 5 clients in tmux and logs outputs

SESSION_NAME="mpeg_federated"
LOG_DIR="./logs"

# Create log directory
mkdir -p $LOG_DIR

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -n "superlink"

# 1️⃣ Start Flower SuperLink
tmux send-keys -t $SESSION_NAME:superlink \
"flower-superlink --insecure | tee $LOG_DIR/superlink.log; read" C-m

sleep 2  # Wait 2 seconds for SuperLink to start

# 2️⃣ Start 5 SuperNodes (one per fold_id)
NUM_CLIENTS=5
SUPERLINK_ADDR="127.0.0.1:9092"
BASE_CLIENTAPI_PORT=9094

for ((fold_id=0; fold_id<$NUM_CLIENTS; fold_id++)); do
    WINDOW_NAME="supernode_$fold_id"
    CLIENTAPI_PORT=$((BASE_CLIENTAPI_PORT + fold_id))
    LOG_FILE="$LOG_DIR/supernode_$fold_id.log"

    tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"
    
    tmux send-keys -t $SESSION_NAME:"$WINDOW_NAME" \
"flower-supernode --insecure \
  --superlink $SUPERLINK_ADDR \
  --clientappio-api-address 127.0.0.1:$CLIENTAPI_PORT \
  --node-config \"fold_id=$fold_id num-partitions=$NUM_CLIENTS\" | tee $LOG_FILE; read" C-m

    sleep 2  # Wait 2 seconds between starting each SuperNode
done

# 3️⃣ Start local-federation run
tmux new-window -t $SESSION_NAME -n "local_federation"
tmux send-keys -t $SESSION_NAME:local_federation \
"flwr run . local-federation --stream | tee $LOG_DIR/local_federation.log; read" C-m

# 4️⃣ Attach to the tmux session
tmux attach -t $SESSION_NAME
