@echo off
cd /d "%~dp0\.."

python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness main --first-order-pretrain-only --pretrain-epochs 6500 --noise 0 --measurements 1501 --outdir outputs\first_order_weak_sigmoid_main
if errorlevel 1 exit /b %errorlevel%

python src\mvm_pinn_jera1.py --stiffness-profile sigmoid --relative-formulation first_order --first-order-physics weak --sigmoid-robustness main --relative-state-checkpoint outputs\first_order_weak_sigmoid_main\relative_state_pretrained_sigmoid.pt --epochs 6000 --noise 0 --measurements 1501 --sigmoid-seeds 2026 2027 2028 --outdir outputs\first_order_weak_sigmoid_main
