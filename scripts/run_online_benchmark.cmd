@echo off
cd /d "%~dp0\.."

python src\mvm_pinn_jera1.py --online-benchmark --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --relative-state-checkpoint outputs\first_order_weak_sigmoid_main\relative_state_pretrained_sigmoid.pt --online-strides 50 --online-adam-steps 5 --noise 0 --measurements 1501 --outdir outputs\online_benchmark
