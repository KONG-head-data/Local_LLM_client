#!/bin/bash

# Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Find all .gguf files
mapfile -t models < <(find models/ -name "*.gguf")

if [ ${#models[@]} -eq 0 ]; then
    echo "Error: No .gguf models found in models/ directory."
    exit 1
fi

# Selection menu logic
selected=0
num_models=${#models[@]}

# Function to draw the menu
draw_menu() {
    for i in "${!models[@]}"; do
        if [ "$i" -eq "$selected" ]; then
            printf "\e[1;32m> %s\e[0m\n" "${models[$i]}"
        else
            printf "  %s\n" "${models[$i]}"
        fi
    done
}

# Function to clear the menu
clear_menu() {
    for ((i=0; i<num_models; i++)); do
        printf "\e[A\e[K"
    done
}

echo "Use Up/Down arrows to select a model and press Enter:"
draw_menu

while true; do
    read -rsn1 key
    case "$key" in
        $'\e') # Escape sequence
            read -rsn2 key
            if [ "$key" == "[A" ]; then # Up arrow
                clear_menu
                ((selected--))
                [ $selected -lt 0 ] && selected=$((num_models - 1))
                draw_menu
            elif [ "$key" == "[B" ]; then # Down arrow
                clear_menu
                ((selected++))
                [ $selected -ge $num_models ] && selected=0
                draw_menu
            fi
            ;;
        "") # Enter key
            break
            ;;
    esac
done

MODEL_PATH="${models[$selected]}"
echo -e "\nStarting llama.cpp server with model: \e[1;34m$MODEL_PATH\e[0m"

# Export for sub-processes if needed (like the API)
export MODEL_PATH="$MODEL_PATH"

# Start the llama.cpp server
python -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 9090 \
  --n_gpu_layers -1 \
  --n_ctx 16384 \
  --flash_attn true \
  --n_batch 64 \
  --n_ubatch 64 \
  --logits_all false
